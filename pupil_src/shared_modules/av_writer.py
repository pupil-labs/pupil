"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

"""
av_writer module uses PyAV (ffmpeg or libav backend) to write AV files.
requires:
    -
"""

# logging
import logging
import multiprocessing as mp
import os
import platform
import sys
from fractions import Fraction
from threading import Event, Thread
from time import time
import abc
import typing as T

import numpy as np

import audio_utils
import av
from av.packet import Packet

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
    ts_file = "{}_timestamps".format(name)
    ts_loc = os.path.join(directory, ts_file)
    ts = np.array(timestamps)
    if output_format not in ("npy", "csv", "all"):
        raise ValueError("Unknown timestamp output format `{}`".format(output_format))
    if output_format in ("npy", "all"):
        np.save(ts_loc + ".npy", ts)
    if output_format in ("csv", "all"):
        np.savetxt(ts_loc + ".csv", ts, fmt="%f", header="timestamps [seconds]")


class AV_Writer(abc.ABC):
    def __init__(
        self, output_file_path: str, frame_pts_timebase: T.Optional[Fraction] = None
    ):
        """
        A generic writer for frames to a file using pyAV.

        output_file_path: File to write frames to.

        frame_pts_timebase:
            If set to None, pts of written frames will be calculated from their timestamp.
            If set, frames are expected to have correct pts for the given timebase.
                The output stream will be created with this timebase.
        """

        self.timestamps = []
        self.frame_pts_timebase = frame_pts_timebase
        self.last_pts = float("-inf")

        self.output_file_path = output_file_path
        directory, video_file = os.path.split(output_file_path)
        name, ext = os.path.splitext(video_file)

        if ext not in self.supported_extensions:
            logger.warning(
                f"Opening media file writer for {ext}. "
                f"Only {self.supported_extensions} are supported! "
                "Using a different container is risky!"
            )

        self.container = av.open(output_file_path, "w")
        logger.debug("Opened '{}' for writing.".format(output_file_path))

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
    def time_base(self) -> Fraction:
        """The desired time base for the output stream."""

        if self.frame_pts_timebase is not None:
            return self.frame_pts_timebase
        # fallback to highest resolution for mp4
        return Fraction(1, 65535)

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
            self.start_time = input_frame.timestamp
            self.configured = True
            self.on_first_frame(input_frame)

        ts = input_frame.timestamp
        if self.timestamps:
            last_ts = self.timestamps[-1]
            if ts < last_ts:
                self.release()
                raise ValueError(
                    "Non-monotonic timestamps!"
                    f"Last timestamp: {last_ts}. Given timestamp: {ts}"
                )

        if self.frame_pts_timebase is not None:
            pts = input_frame.pts
        else:
            # need to calculate frame pts based on timestamp
            pts = int((input_frame.timestamp - self.start_time) / self.time_base)

        # ensure strong monotonic pts
        pts = max(pts, self.last_pts + 1)
        self.last_pts = pts

        for packet in self.encode_frame(input_frame, pts):
            self.container.mux(packet)

        self.timestamps.append(ts)

    def close(self, timestamp_export_format="npy"):
        """Close writer, triggering stream and timestamp save."""

        if self.closed:
            logger.warning("Trying to close container multiple times!")
            return

        # cannot close container, if no frames were written (i.e. no timestamps recorded)
        if not self.timestamps:
            logger.warning("Trying to close container without any frames written!")
            for packet in self.video_stream.encode(None):
                self.container.mux(packet)
        self.container.close()

        write_timestamps(
            self.output_file_path, self.timestamps, timestamp_export_format
        )
        self.closed = True

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
        """Supported file extensions."""

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
        self.frame.dts = pts

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
        packet = Packet()
        packet.payload = input_frame.jpeg_buffer
        packet.time_base = self.time_base
        packet.dts = pts
        packet.pts = pts
        yield packet


class MPEG_Audio_Writer(MPEG_Writer):
    """Extension of MPEG_Writer with audio support."""

    def __init__(
        self,
        output_file_path: str,
        audio_dir: str,
        **kwargs
    ):
        super().__init__(output_file_path, **kwargs)

        try:
            self.audio = audio_utils.load_audio(audio_dir)
            self.audio_export_stream = self.container.add_stream(
                template=self.audio.stream
            )
        except audio_utils.NoAudioLoadedError:
            logger.warning("Could not mux audio. File not found.")
            self.audio = None

        self.num_audio_packets_decoded = 0

    def encode_frame(self, input_frame, pts: int) -> T.Iterator[Packet]:
        # encode video packets from AV_Writer base first
        yield from super().encode_frame(input_frame, pts)

        if self.audio is None:
            return

        # encode audio packets
        for audio_packet in self.audio.container.demux():
            if self.num_audio_packets_decoded >= len(self.audio.timestamps):
                logger.debug(
                    "More audio frames decoded than there are timestamps: {} > {}".format(
                        self.num_audio_packets_decoded, len(self.audio.timestamps)
                    )
                )
                break
            audio_pts = int(
                (
                    self.audio.timestamps[self.num_audio_packets_decoded]
                    - self.start_time
                )
                / self.audio_export_stream.time_base
            )
            audio_packet.pts = audio_pts
            audio_packet.dts = audio_pts
            audio_packet.stream = self.audio_export_stream
            self.num_audio_packets_decoded += 1

            audio_ts = audio_pts * self.audio_export_stream.time_base
            if audio_ts < 0:
                logger.debug("Seeking: {} -> {}".format(audio_ts, self.start_time))
                continue  # seek to start_time

            yield audio_packet

            frame_ts = self.frame.pts * self.time_base
            if audio_ts > frame_ts:
                break  # wait for next image


def format_time(time, time_base):
    if time is None:
        return "None"
    return "{:.3f}s ({} or {}/{})".format(
        time_base * time,
        time_base * time,
        time_base.numerator * time,
        time_base.denominator,
    )


def rec_thread(file_loc, in_container, audio_src, should_close):
    # print sys.modules['av']
    # import av
    if not in_container:
        # create in container
        if platform.system() == "Darwin":
            in_container = av.open("none:{}".format(audio_src), format="avfoundation")
        elif platform.system() == "Linux":
            in_container = av.open("hw:{}".format(audio_src), format="alsa")

    in_stream = None

    # print len(in_container.streams), 'stream(s):'
    for i, stream in enumerate(in_container.streams):

        if stream.type == "audio":
            # print '\t\taudio:'
            # print '\t\t\tformat:', stream.format
            # print '\t\t\tchannels: %s' % stream.channels
            in_stream = stream
            break

    if in_stream is None:
        # logger.error("No input audio stream found.")
        return

    # create out container
    out_container = av.open(file_loc, "w")
    # logger.debug("Opened '%s' for writing."%file_loc)
    out_stream = out_container.add_stream(template=in_stream)

    for packet in in_container.demux(in_stream):
        # for frame in packet.decode():
        #     packet = out_stream.encode(frame)
        #     if packet:
        # print '%r' %packet
        # print '\tduration: %s' % format_time(packet.duration, packet.stream.time_base)
        # print '\tpts: %s' % format_time(packet.pts, packet.stream.time_base)
        # print '\tdts: %s' % format_time(packet.dts, packet.stream.time_base)
        out_container.mux(packet)
        if should_close.is_set():
            break

    out_container.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from audio_capture import Audio_Capture

    cap = Audio_Capture("test.wav", "default")

    import time

    time.sleep(5)
    cap.cleanup()
    # mic device
    exit()

    # container = av.open('hw:0',format="alsa")
    container = av.open("1:0", format="avfoundation")
    print("container:", container)
    print("\tformat:", container.format)
    print("\tduration:", float(container.duration) / av.time_base)
    print("\tmetadata:")
    for k, v in sorted(container.metadata.items()):
        print("\t\t{}: {!r}".format(k, v))

    print(len(container.streams), "stream(s):")
    audio_stream = None
    for i, stream in enumerate(container.streams):

        print("\t{!r}".format(stream))
        print("\t\ttime_base: {!r}".format(stream.time_base))
        print("\t\trate: {!r}".format(stream.rate))
        print("\t\tstart_time: {!r}".format(stream.start_time))
        print("\t\tduration: {}".format(format_time(stream.duration, stream.time_base)))
        print("\t\tbit_rate: {}".format(stream.bit_rate))
        print("\t\tbit_rate_tolerance: {}".format(stream.bit_rate_tolerance))

        if stream.type == b"audio":
            print("\t\taudio:")
            print("\t\t\tformat:", stream.format)
            print("\t\t\tchannels: {}".format(stream.channels))
            audio_stream = stream
            break
        elif stream.type == "container":
            print("\t\tcontainer:")
            print("\t\t\tformat:", stream.format)
            print("\t\t\taverage_rate: {!r}".format(stream.average_rate))

        print("\t\tmetadata:")
        for k, v in sorted(stream.metadata.items()):
            print("\t\t\t{}: {!r}".format(k, v))

    if not audio_stream:
        exit()

    # file contianer:
    out_container = av.open("test.wav", "w")
    out_stream = out_container.add_stream(template=audio_stream)
    # out_stream.rate = 44100
    for i, packet in enumerate(container.demux(audio_stream)):
        # for frame in packet.decode():
        #     packet = out_stream.encode(frame)
        #     if packet:
        print("{!r}".format(packet))
        print(
            "\tduration: {}".format(
                format_time(packet.duration, packet.stream.time_base)
            )
        )
        print("\tpts: {}".format(format_time(packet.pts, packet.stream.time_base)))
        print("\tdts: {}".format(format_time(packet.dts, packet.stream.time_base)))
        out_container.mux(packet)
        if i > 1000:
            break

    out_container.close()

    # import cProfile,subthread,os
    # cProfile.runctx("test()",{},locals(),"av_writer.pstats")
    # loc = os.path.abspath(_file__).rsplit('pupil_src', 1)
    # gprof2dot_loc = os.path.oin(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    # subthread.call("python "+gprof2dot_loc+" -f pstats av_writer.pstats | dot -Tpng -o av_writer.png", shell=True)
    # print "created cpu time graph for av_writer thread. Please check out the png next to the av_writer.py file"
