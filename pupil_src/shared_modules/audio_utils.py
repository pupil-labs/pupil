"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import logging
import traceback
import typing as T

import av
import numpy as np
import pupil_recording

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NoAudioLoadedError(Exception):
    pass


class LoadedAudio(T.NamedTuple):
    container: T.Any
    stream: T.Any
    timestamps: T.List[float]
    pts: T.List[int]

    def __str__(self):
        return (
            f"{type(self).__name__}(container={self.container}, stream={self.stream}, "
            f"timestamps=(N={len(self.timestamps)}, [{self.timestamps[0]}, "
            f"{self.timestamps[-1]}]), pts=(N={len(self.pts)}, [{self.pts[0]}, "
            f"{self.pts[-1]}]))"
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
        logger.debug(f"Loaded audiostream: {stream} from {file_path}")
    except (av.AVError, StopIteration):
        return None

    ts_path = file_path.with_name(file_path.stem + "_timestamps.npy")
    try:
        timestamps = np.load(ts_path)
    except OSError:
        return None

    start = timestamps[0]
    packet_pts = np.array(
        [p.pts for p in container.demux(stream) if p is not None and p.pts is not None],
    )

    if return_pts_based_timestamps:
        timestamps = start + packet_pts * stream.time_base

    # pts seeking requires primitive Python integers and does not accept numpy int types;
    # `.tolist()` converts numpy integers to primitive Python integers; do conversion after
    # `packet_pts * stream.time_base` to leverage numpy element-wise function application
    packet_pts = packet_pts.tolist()

    try:
        container.seek(0)
    except av.AVError as err:
        logger.debug(f"{err}")
        return None

    return LoadedAudio(container, stream, timestamps, packet_pts)


class Audio_Viz_Transform:
    def __init__(self, rec_dir, log_scaling=False, sps_rate=60):
        logger.debug("Audio_Viz_Transform.__init__: Loading audio")
        self.audio_all = iter(load_audio(rec_dir))
        self._setup_next_audio_part()
        self._first_part_start = self.audio.timestamps[0]

        self.sps_rate = sps_rate
        self.all_abs_samples = None
        self.finished = False
        self.a_levels = None
        self.a_levels_log = None
        self.final_rescale = True
        self.log_scaling = log_scaling

    def _setup_next_audio_part(self):
        self.audio = next(self.audio_all)
        logger.debug(
            f"Audio_Viz_Transform._setup_next_audio_part: Part {self.audio.container} {self.audio.stream}"
        )
        self.audio_resampler = av.audio.resampler.AudioResampler(
            format=self.audio.stream.format, layout=self.audio.stream.layout, rate=60
        )
        logger.debug(
            "Audio_Viz_Transform._setup_next_audio_part: Resampler initialized"
        )
        self.next_audio_frame = self._next_audio_frame()
        self.start_ts = self.audio.timestamps[0]

    def _next_audio_frame(self):
        for packet in self.audio.container.demux(self.audio.stream):
            try:
                for frame in packet.decode():
                    if frame:
                        yield frame
            except av.AVError:
                logger.debug(traceback.format_exc())

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
                try:
                    audio_frames_rs = self.audio_resampler.resample(audio_frame)
                except av.error.ValueError:
                    continue
                if not audio_frames_rs:
                    continue
                samples = np.concatenate(
                    [frame.to_ndarray().ravel() for frame in audio_frames_rs], axis=0
                )
                if allSamples is not None:
                    allSamples = np.concatenate((allSamples, samples), axis=0)
                else:
                    allSamples = samples

            # flush
            audio_frames_rs = self.audio_resampler.resample(None)
            if audio_frames_rs:
                samples = np.concatenate(
                    [frame.to_ndarray().ravel() for frame in audio_frames_rs], axis=0
                )
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

            elif self.a_levels is not None and self.all_abs_samples is not None:
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
            else:
                logger.debug(
                    f"Audio_Viz_Transform.get_data: No audio found in {self.audio}"
                )
                new_ts = None
                try:
                    self._setup_next_audio_part()
                except StopIteration:
                    self.finished = True

            if new_ts is not None and (not self.finished or self.final_rescale):
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
