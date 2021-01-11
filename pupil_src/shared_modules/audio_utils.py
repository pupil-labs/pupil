"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import logging

import av
import numpy as np

import pupil_recording

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NoAudioLoadedError(Exception):
    pass


LoadedAudio = collections.namedtuple(
    "LoadedAudio", ["container", "stream", "timestamps"]
)


def load_audio(rec_dir):
    recording = pupil_recording.PupilRecording(rec_dir)
    loaded_audio = _load_audio_from_audio_files(recording)
    if not loaded_audio:
        loaded_audio = _load_audio_from_world_video_files(recording)
    if not loaded_audio:
        raise NoAudioLoadedError("No valid audio file found")
    return loaded_audio


def _load_audio_from_audio_files(recording):
    audio_files = sorted(recording.files().audio().mp4())
    loaded_audio = (_load_audio_single(path) for path in audio_files)
    loaded_audio = [aud for aud in loaded_audio if aud is not None]
    return loaded_audio


def _load_audio_from_world_video_files(recording):
    audio_files = sorted(recording.files().world().videos())
    loaded_audio = (
        _load_audio_single(path, return_pts_based_timestamps=True)
        for path in audio_files
    )
    loaded_audio = [aud for aud in loaded_audio if aud is not None]

    return loaded_audio


def _load_audio_single(file_path, return_pts_based_timestamps=False):
    try:
        container = av.open(str(file_path))
        stream = next(iter(container.streams.audio))
        logger.debug("Loaded audiostream: %s" % stream)
    except (av.AVError, StopIteration):
        return None

    ts_path = file_path.with_name(file_path.stem + "_timestamps.npy")
    try:
        timestamps = np.load(ts_path)
    except IOError:
        return None

    if return_pts_based_timestamps:
        start = timestamps[0]
        timestamps = np.fromiter(
            (
                float(start + p.pts * p.time_base)
                for p in container.demux(audio=0)
                if p is not None and p.pts is not None and p.time_base is not None
            ),
            dtype=float,
        )
        try:
            container.seek(0)
        except av.AVError as err:
            logger.debug(f"{err}")
            return None

    return LoadedAudio(container, stream, timestamps)


class Audio_Viz_Transform:
    def __init__(self, rec_dir, sps_rate=60):
        self.audio_all = iter(load_audio(rec_dir))
        self._setup_next_audio_part()
        self._first_part_start = self.audio.timestamps[0]

        self.sps_rate = sps_rate
        self.all_abs_samples = None
        self.finished = False
        self.a_levels = None
        self.a_levels_log = None
        self.final_rescale = True
        self.log_scaling = False

    def _setup_next_audio_part(self):
        self.audio = next(self.audio_all)
        self.audio_resampler = av.audio.resampler.AudioResampler(
            format=self.audio.stream.format, layout=self.audio.stream.layout, rate=60
        )
        self.next_audio_frame = self._next_audio_frame()
        self.start_ts = self.audio.timestamps[0]

    def _next_audio_frame(self):
        for packet in self.audio.container.demux(self.audio.stream):
            for frame in packet.decode():
                if frame:
                    yield frame

    def sec_to_frames(self, sec):
        return int(np.ceil(sec * self.audio.stream.rate / self.audio.stream.frame_size))

    def get_data(self, seconds=30.0, height=210, log_scale=False):
        import itertools

        finished = self.finished
        if not finished:
            allSamples = None
            frames_to_fetch = self.sec_to_frames(seconds)
            if frames_to_fetch > 0:
                frames_chunk = itertools.islice(self.next_audio_frame, frames_to_fetch)
            for af in frames_chunk:
                audio_frame = af
                audio_frame.pts = None
                # af_dbl = audio_resampler1.resample(af)
                # lp_graph_list[0].push(af)
                # audio_frame = lp_graph_list[-1].pull()
                # if audio_frame is None:
                #    continue
                # audio_frame.pts = None
                audio_frame_rs = self.audio_resampler.resample(audio_frame)
                if audio_frame_rs is None:
                    continue
                samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
                if allSamples is not None:
                    allSamples = np.concatenate((allSamples, samples), axis=0)
                else:
                    allSamples = samples

            # flush
            audio_frame_rs = self.audio_resampler.resample(None)
            if audio_frame_rs is not None:
                samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
                if allSamples is not None:
                    allSamples = np.concatenate((allSamples, samples), axis=0)
                else:
                    allSamples = samples
            if allSamples is not None:
                new_ts = (
                    np.arange(0, len(allSamples), 1, dtype=np.float32)
                    / self.audio_resampler.rate
                )
                new_ts += self.start_ts
                self.start_ts = new_ts[-1] + 1 / self.audio_resampler.rate

                abs_samples = np.abs(allSamples)
                if self.all_abs_samples is not None:
                    self.all_abs_samples = np.concatenate(
                        (self.all_abs_samples, abs_samples), axis=0
                    )
                else:
                    self.all_abs_samples = abs_samples

                scaled_samples_log = self.log_scale(abs_samples)

                if abs_samples.max() - abs_samples.min() > 0.0:
                    scaled_samples = (abs_samples - abs_samples.min()) / (
                        abs_samples.max() - abs_samples.min()
                    )
                elif abs_samples.max() > 0.0:
                    scaled_samples = abs_samples / abs_samples.max()
                else:
                    scaled_samples = abs_samples

            else:
                new_ts = self.a_levels[::4]  # reconstruct correct ts

                # self.all_abs_samples = np.log10(self.all_abs_samples)
                self.all_abs_samples[-1] = 0.0

                scaled_samples_log = self.log_scale(self.all_abs_samples)

                if self.all_abs_samples.max() - self.all_abs_samples.min() > 0.0:
                    scaled_samples = (
                        self.all_abs_samples - self.all_abs_samples.min()
                    ) / (self.all_abs_samples.max() - self.all_abs_samples.min())
                elif self.all_abs_samples.max() > 0.0:
                    scaled_samples = self.all_abs_samples / self.all_abs_samples.max()
                else:
                    scaled_samples = self.all_abs_samples
                self.a_levels = None

                try:
                    self._setup_next_audio_part()
                except StopIteration:
                    self.finished = True
            if not self.finished or self.final_rescale:
                a_levels = self.get_verteces(new_ts, scaled_samples, height)

                if self.a_levels is not None:
                    self.a_levels = np.concatenate((self.a_levels, a_levels), axis=0)
                else:
                    self.a_levels = a_levels

                a_levels_log = self.get_verteces(new_ts, scaled_samples_log, height)

                if self.a_levels_log is not None:
                    self.a_levels_log = np.concatenate(
                        (self.a_levels_log, a_levels_log), axis=0
                    )
                else:
                    self.a_levels_log = a_levels_log

        if not log_scale:
            ret = self.a_levels
        else:
            ret = self.a_levels_log

        if self.log_scaling != log_scale:
            self.log_scaling = log_scale
            finished = False

        return ret, finished

    def get_verteces(self, new_ts, scaled_samples, height):
        points_y1 = scaled_samples * (-height / 2) + height / 2
        points_xy1 = np.concatenate(
            (new_ts.reshape(-1, 1), points_y1.reshape(-1, 1)), 1
        )
        points_y2 = scaled_samples * (height / 2) + height / 2
        points_xy2 = np.concatenate(
            (new_ts.reshape(-1, 1), points_y2.reshape(-1, 1)), 1
        )
        # a_levels = [alevel for alevel in zip(new_ts, scaled_samples)]
        a_levels = np.concatenate((points_xy1, points_xy2), 1).reshape(-1)

        return a_levels

    def log_scale(self, abs_samples):
        scaled_samples = abs_samples / abs_samples.max() + 0.0001
        scaled_samples_log = 10 * np.log10(scaled_samples)
        sc_min = scaled_samples_log.min()
        scaled_samples_log += -sc_min
        sc_max = scaled_samples_log.max()
        scaled_samples_log /= sc_max

        return scaled_samples_log
