"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import collections
import logging
import math
import multiprocessing as mp
import os
import typing as T
from fractions import Fraction

import audio_utils
import av
import numpy as np
from av.packet import Packet
from methods import container_decode, iter_catch
from video_capture.utils import InvalidContainerError, Video

logger = logging.getLogger(__name__)


"""
notes on time_bases and presentation timestamps:

Time_base (fraction) is the conversion factor to multipy the uint64 pts value to seconds

two time bases that we care about exsist:
    time_base of the stream (AVStream) this is used for the packet pts/dts
    time_base fo the codec (AVCodecContext) this is used for the frame

going from packet pts to frame pts when decoding:
frame.pts  = av_rescale_q ( packet. pts , packetTimeBase , frameTimeBase )

..when encoding:
packet.pts  = av_rescale_q ( frame. pts , frameTimeBase,packetTimeBase  )


Setting the time_base:
The timebase of the codec is settable (and only settable at the beginnig):
currently in PyAV this is done container.add_stream(codec,codec_timebase)

The timebase of the stream is not user settable. It is determined by ffmpeg.
The streamtimebase uses the codec timebase as a hint to find a good value.
The stream timebase in influenced by the contraints/rules of the container as well.
Only when the header  of the stream is written stream.time_base is garanteed
to be valid and should only now be accesed.
"""


def write_timestamps(file_loc, timestamps, output_format="npy"):
    """
    Attritbutes:
        output_format (str): Output file format. Available values:
        - "npy": numpy array format (default)
        - "csv": csv file export
        - "all": Exports in all of the above formats
    """
    directory, video_file = os.path.split(file_loc)
    name, ext = os.path.splitext(video_file)
    ts_file = f"{name}_timestamps"
    ts_loc = os.path.join(directory, ts_file)
    ts = np.array(timestamps)
    if output_format not in ("npy", "csv", "all"):
        raise ValueError(f"Unknown timestamp output format `{output_format}`")
    if output_format in ("npy", "all"):
        np.save(ts_loc + ".npy", ts)
    if output_format in ("csv", "all"):
        output_video = Video(file_loc)
        try:
            container = output_video.load_container()
            pts = output_video.load_pts(container)
            ts_pts = np.vstack((ts, pts)).T
            np.savetxt(
                ts_loc + ".csv",
                ts_pts,
                fmt=["%f", "%i"],
                delimiter=",",
                header="timestamps [seconds],pts",
            )
        except InvalidContainerError:
            logger.error(f"Failed to extract PTS frome exported video {file_loc}")
            return


class NonMonotonicTimestampError(ValueError):
    """Indicates that a Writer received non-monotonic data to write."""

    pass


class AV_Writer(abc.ABC):
    def __init__(self, output_file_path: str, start_time_synced: int):
        """
        A generic writer for frames to a file using pyAV.

        output_file_path: File to write frames to.
        start_time_synced: Start time of the recording.
            Will be used to calculate positions of frames (pts).
        """

        self.timestamps = []
        self.start_time = start_time_synced
        self.last_video_pts = float("-inf")

        # always write with highest resolution for mp4
        # NOTE: libav might lower the resolution on saving, if possible
        self.time_base = Fraction(1, 65535)

        self.output_file_path = output_file_path
        directory, video_file = os.path.split(output_file_path)
        name, ext = os.path.splitext(video_file)

        if ext not in self.supported_extensions:
            logger.warning(
                f"Opening media file writer for {ext}. "
                f"Only {self.supported_extensions} are supported! "
                "Using a different container is risky!"
            )

        # `ext` starts with a dot, so we need to remove it for the format to be
        # recongnized by pyav
        self.container = av.open(self.output_file_path_in_porgress, "w", format=ext[1:])
        logger.debug(f"Opened '{output_file_path}' for writing.")

        self.configured = False
        self.video_stream = self.container.add_stream(
            codec_name=self.codec, rate=1 / self.time_base
        )

        # TODO: Where does this bit-rate come from? Seems like an unreasonable
        # value. Also 10e3 == 1e4, which makes this even weirder!
        BIT_RATE = 15000 * 10e3
        self.video_stream.bit_rate = BIT_RATE
        self.video_stream.bit_rate_tolerance = BIT_RATE / 20
        self.video_stream.thread_count = max(1, mp.cpu_count() - 1)

        self.closed = False

    @property
    def output_file_path_in_porgress(self):
        """
        Return the output file path, with a suffix to indicate that writing is in progress.
        """
        return self.output_file_path + ".writing"

    def write_video_frame(self, input_frame):
        """
        Write a frame to the video_stream.

        For subclasses, implement self.encode_frame().
        """
        if self.closed:
            logger.warning("Container was closed already!")
            return

        if not self.configured:
            self.video_stream.height = input_frame.height
            self.video_stream.width = input_frame.width
            self.configured = True
            self.on_first_frame(input_frame)

        ts = input_frame.timestamp

        if ts < self.start_time:
            # This can happen, because we might have a frame already in the
            # pipeline when starting the recording. We should skip this frame
            # then, as the processes are not yet synced.
            logger.debug("Skipping frame that arrived before sync time.")
            return

        if self.timestamps:
            last_ts = self.timestamps[-1]
            if ts < last_ts:
                self.release()
                raise NonMonotonicTimestampError(
                    "Non-monotonic timestamps!"
                    f"Last timestamp: {last_ts}. Given timestamp: {ts}"
                )

        pts = int((input_frame.timestamp - self.start_time) / self.time_base)

        # ensure strong monotonic pts
        pts = max(pts, self.last_video_pts + 1)

        # TODO: Use custom Frame wrapper class, that wraps backend-specific frames.
        # This way we could just attach the pts here to the frame.
        # Currently this will fail e.g. for av.VideoFrame.
        video_packed_encoded = False
        for packet in self.encode_frame(input_frame, pts):
            if packet.stream is self.video_stream:
                if video_packed_encoded:
                    # NOTE: Assumption: Each frame is encoded into a single packet!
                    # This is required for the frame.pts == packet.pts assumption below.
                    logger.warning("Single frame yielded more than one packet")
                video_packed_encoded = True
            self.container.mux(packet)

        if not video_packed_encoded:
            logger.warning(f"Encoding frame {input_frame.index} failed!")
            return

        self.last_video_pts = pts
        self.timestamps.append(ts)

    def close(self, timestamp_export_format="npy", closed_suffix=""):
        """Close writer, triggering stream and timestamp save.

        closed_suffix: Use to indicate a cancelled or failed write.
        """

        if self.closed:
            logger.warning("Trying to close container multiple times!")
            return

        if self.configured:
            # at least one frame has been written, flush stream
            for packet in self.video_stream.encode(None):
                self.container.mux(packet)

        self.container.close()
        self.closed = True

        output_file_path = self.output_file_path + closed_suffix
        os.rename(self.output_file_path_in_porgress, output_file_path)

        if self.configured and timestamp_export_format is not None:
            # Requires self.container to be closed since we extract pts
            # from the exported video file.
            write_timestamps(output_file_path, self.timestamps, timestamp_export_format)

    def release(self):
        """Close writer, triggering stream and timestamp save."""
        self.close()

    def on_first_frame(self, input_frame) -> None:
        """
        Will be called once for the first frame.

        Overwrite to do additional setup.
        """
        pass

    @abc.abstractmethod
    def encode_frame(self, input_frame, pts: int) -> T.Iterator[Packet]:
        """Encode a frame into one or multiple av packets with given pts."""

    @property
    @abc.abstractmethod
    def supported_extensions(self) -> T.Tuple[str]:
        """Supported file extensions (starting with '.')."""

    @property
    @abc.abstractmethod
    def codec(self) -> str:
        """Desired video stream codec."""


class MPEG_Writer(AV_Writer):
    """AV_Writer with MPEG4 encoding."""

    @property
    def supported_extensions(self):
        return (".mp4", ".mov", ".mkv")

    @property
    def codec(self):
        return "mpeg4"

    def on_first_frame(self, input_frame) -> None:
        # setup av frame once to use as buffer throughout the process
        if input_frame.yuv_buffer is not None:
            pix_format = "yuv422p"
        else:
            pix_format = "bgr24"
        self.frame = av.VideoFrame(input_frame.width, input_frame.height, pix_format)
        self.frame.time_base = self.time_base

    def encode_frame(self, input_frame, pts: int) -> T.Iterator[Packet]:
        if input_frame.yuv_buffer is not None:
            y, u, v = input_frame.yuv422
            self.frame.planes[0].update(y)
            self.frame.planes[1].update(u)
            self.frame.planes[2].update(v)
        else:
            self.frame.planes[0].update(input_frame.img)

        self.frame.pts = pts

        yield from self.video_stream.encode(self.frame)


class JPEG_Writer(AV_Writer):
    """AV_Writer with MJPEG encoding."""

    @property
    def supported_extensions(self):
        return (".mp4",)

    @property
    def codec(self):
        return "mjpeg"

    def on_first_frame(self, input_frame) -> None:
        self.video_stream.pix_fmt = "yuvj422p"

    def encode_frame(self, input_frame, pts: int) -> T.Iterator[Packet]:
        # for JPEG we only get a single packet per frame
        try:
            packet = Packet()
            packet.payload = input_frame.jpeg_buffer
        except AttributeError:
            packet = Packet(input_frame.jpeg_buffer)
            # packet.update()
        packet.stream = self.video_stream
        packet.time_base = self.time_base
        packet.pts = pts
        # TODO: check if we still need dts here, as they were removed from MPEG_Writer
        packet.dts = pts
        yield packet


class MPEG_Audio_Writer(MPEG_Writer):
    """Extension of MPEG_Writer with audio support."""

    @staticmethod
    def _add_stream(container, template):
        stream = container.add_stream(
            codec_name=template.codec.name, rate=template.rate
        )
        try:
            stream.layout = template.layout
        except AttributeError:
            pass
        return stream

    def __init__(self, *args, audio_dir: str, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.audio_parts = audio_utils.load_audio(audio_dir)
            self.audio_export_stream = type(self)._add_stream(
                container=self.container, template=self.audio_parts[0].stream
            )
        except audio_utils.NoAudioLoadedError:
            logger.debug("Could not mux audio. File not found.")
            self.audio_parts = None
            return

        # setup stateful packet iterator
        self.audio_packet_iterator = self.iterate_audio_packets()

    def encode_frame(self, input_frame, pts: int) -> T.Iterator[Packet]:
        # encode video packets from AV_Writer base first
        yield from super().encode_frame(input_frame, pts)

        if self.audio_parts is None:
            return

        # encode all audio packets up to current frame timestamp
        frame_ts = self.frame.pts * self.time_base
        for audio_packet in self.audio_packet_iterator:
            audio_ts = audio_packet.pts * self.audio_export_stream.time_base

            yield audio_packet

            if audio_ts > frame_ts:
                # done for this frame, pause iteration
                return

    def iterate_audio_packets(self):
        """Yields all audio packets from start_time to end."""
        yield from _AudioPacketIterator(
            start_time=self.start_time,
            audio_export_stream=self.audio_export_stream,
            audio_parts=self.audio_parts,
            fill_gaps=True,
        ).iterate_audio_packets()


class _AudioPacketIterator:
    def __init__(self, start_time, audio_parts, audio_export_stream, fill_gaps=True):
        self.start_time = start_time
        self.audio_parts = audio_parts
        self.audio_export_stream = audio_export_stream
        self.fill_gaps = fill_gaps

    def iterate_audio_packets(self):

        last_audio_pts = float("-inf")

        if self.fill_gaps:
            audio_frames_iterator = self._iterate_audio_frames_filling_gaps()
        else:
            audio_frames_iterator = self._iterate_audio_frames_ignoring_gaps()

        for audio_frame in audio_frames_iterator:
            frame, raw_ts = audio_frame.raw_frame, audio_frame.start_time

            try:
                for packet in self.audio_export_stream.encode(frame):
                    if not packet:
                        continue

                    audio_ts = raw_ts - self.start_time
                    audio_pts = int(audio_ts / self.audio_export_stream.time_base)

                    # ensure strong monotonic pts
                    audio_pts = max(audio_pts, last_audio_pts + 1)
                    last_audio_pts = audio_pts

                    packet.pts = audio_pts
                    packet.dts = audio_pts
                    packet.stream = self.audio_export_stream

                    if audio_ts < 0:
                        logger.debug(f"Seeking audio: {audio_ts} -> {self.start_time}")
                        # discard all packets before start time
                        return None

                    yield packet
            except ValueError as exc:
                # TODO: investigate cause
                logger.debug(f"Failed encoding audio frames {frame} due to {exc}")

    # Private

    class _AudioGap(collections.namedtuple("_AudioGap", ["start_time", "end_time"])):
        @property
        def duration(self):
            return self.end_time - self.start_time

    class _AudioFrame(
        collections.namedtuple("_AudioFrame", ["raw_frame", "start_time"])
    ):
        @property
        def duration(self):
            return self.raw_frame.samples / self.raw_frame.sample_rate

        @property
        def end_time(self):
            return self.start_time + self.duration

    def _iterate_audio_frames_ignoring_gaps(self):
        for audio_frame in self._iterate_audio_frames_and_audio_gaps(self.audio_parts):
            if isinstance(audio_frame, _AudioPacketIterator._AudioFrame):
                yield audio_frame
            elif isinstance(audio_frame, _AudioPacketIterator._AudioGap):
                continue
            else:
                raise ValueError(f"Unknown audio frame type: {audio_frame}")

    def _iterate_audio_frames_filling_gaps(self):

        # Prologue: Yield silence frames between start_time and the first audio frame (if any)

        audio_part_end_ts = [part.timestamps[-1] for part in self.audio_parts]
        first_audio_part_idx = np.searchsorted(audio_part_end_ts, self.start_time)
        try:
            first_audio_part = self.audio_parts[first_audio_part_idx]
        except IndexError:
            # export start time is after last audio part, i.e. there is no audio left
            # to export, i.e. no silence needed
            return
        first_audio_part_start_ts = first_audio_part.timestamps[0]
        prologue_duration = first_audio_part_start_ts - self.start_time

        if prologue_duration > 0:
            yield from self._generate_silence_audio_frames(
                stream=self.audio_export_stream,
                start_ts=self.start_time,
                max_duration=prologue_duration,
            )

        # Main part: Yield audio frames, replacing gaps with silence frames

        last_audio_frame = None

        audio_parts = self.audio_parts[first_audio_part_idx:]
        for audio_frame in self._iterate_audio_frames_and_audio_gaps(audio_parts):
            last_audio_frame = audio_frame

            # Skip packets that are before the start_time
            if audio_frame.start_time < self.start_time:
                continue

            if audio_frame.duration <= 0:
                continue

            if isinstance(audio_frame, _AudioPacketIterator._AudioFrame):
                yield audio_frame
            elif isinstance(audio_frame, _AudioPacketIterator._AudioGap):
                yield from self._generate_silence_audio_frames(
                    stream=self.audio_export_stream,
                    start_ts=audio_frame.start_time,
                    max_duration=audio_frame.duration,
                )
            else:
                raise ValueError(f"Unknown audio frame type: {audio_frame}")

        # Epilogue: Infinitely yield silence frames; the client is responsible to stop the iteration

        yield from self._generate_silence_audio_frames(
            stream=self.audio_export_stream,
            start_ts=last_audio_frame.end_time,
            max_duration=None,  # Inifinite generator
        )

    @staticmethod
    def _iterate_audio_frames_and_audio_gaps(audio_parts):
        if audio_parts is None:
            return

        last_part_idx = 0
        last_part_last_frame = None

        for part_idx, audio_part in enumerate(audio_parts):

            frames = container_decode(audio_part.container, audio=0)
            frames = iter_catch(frames, av.AVError)
            for frame, timestamp in zip(frames, audio_part.timestamps):
                if frame is None:
                    continue  # ignore audio decoding errors

                frame.pts = None

                audio_frame = _AudioPacketIterator._AudioFrame(
                    raw_frame=frame, start_time=timestamp
                )

                if part_idx != last_part_idx:
                    audio_gap = _AudioPacketIterator._AudioGap(
                        start_time=last_part_last_frame.end_time,
                        end_time=audio_frame.start_time,
                    )
                    yield audio_gap

                yield audio_frame

                last_part_idx = part_idx
                last_part_last_frame = audio_frame

    @staticmethod
    def _generate_silence_audio_frames(stream, start_ts, max_duration: float = None):

        frame_sample_sizes = _AudioPacketIterator._generate_raw_frame_sample_size(
            stream, start_ts, max_duration
        )
        raw_frame_factory = _AudioPacketIterator._create_audio_raw_frame_factory(stream)
        timestamp_factory = _AudioPacketIterator._create_audio_timestamp_factory(
            stream, start_ts
        )

        for frame_size in frame_sample_sizes:
            frame = raw_frame_factory(frame_size)
            timestamp = timestamp_factory(frame_size)

            audio_frame = _AudioPacketIterator._AudioFrame(
                raw_frame=frame, start_time=timestamp
            )
            yield audio_frame

    @staticmethod
    def _generate_raw_frame_sample_size(
        stream, start_ts: float, max_duration: float = None
    ):
        sample_rate = stream.codec_context.sample_rate
        frame_size = stream.codec_context.frame_size

        def _generate_with_duration(duration: float):
            sample_count = int(sample_rate * duration)
            frame_count = math.floor(sample_count / frame_size)
            for frame_idx in range(frame_count):
                remaining_count = sample_count - (frame_idx * frame_size)
                yield min(remaining_count, frame_size)

        def _generate_infinitely():
            while True:
                yield frame_size

        if max_duration is not None:
            yield from _generate_with_duration(max_duration)
        else:
            yield from _generate_infinitely()

    @staticmethod
    def _create_audio_timestamp_factory(stream, start_ts):
        sample_rate = stream.codec_context.sample_rate
        frame_timestamp = start_ts

        def f(frame_sample_size):
            nonlocal frame_timestamp
            ts = frame_timestamp
            frame_timestamp += frame_sample_size / sample_rate
            return ts

        return f

    @staticmethod
    def _create_audio_raw_frame_factory(stream):
        sample_rate = stream.codec_context.sample_rate
        av_format = stream.codec_context.format.name
        av_layout = stream.codec_context.layout.name
        dtype = np.dtype(_AudioPacketIterator._format_dtypes[av_format])

        def f(frame_sample_size):
            frame = av.AudioFrame(
                samples=frame_sample_size, format=av_format, layout=av_layout
            )
            frame.pts = None
            frame.sample_rate = sample_rate

            for plane in frame.planes:
                buffer = np.frombuffer(plane, dtype=dtype)
                buffer[:] = 0

            return frame

        return f

    # https://github.com/mikeboers/PyAV/blob/develop/av/audio/frame.pyx
    _format_dtypes = {
        "dbl": "<f8",
        "dblp": "<f8",
        "flt": "<f4",
        "fltp": "<f4",
        "s16": "<i2",
        "s16p": "<i2",
        "s32": "<i4",
        "s32p": "<i4",
        "u8": "u1",
        "u8p": "u1",
    }
