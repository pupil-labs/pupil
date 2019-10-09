"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# logging
import logging
import os
import os.path
import av
import numpy as np
import typing as T

from multiprocessing import cpu_count
from abc import ABC, abstractmethod
from time import sleep
from camera_models import load_intrinsics
from .utils import VideoSet

import player_methods as pm
from .base_backend import Base_Manager, Base_Source, EndofVideoError, Playback_Source
from pupil_recording import PupilRecording

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
av.logging.set_level(av.logging.ERROR)
logging.getLogger("libav").setLevel(logging.ERROR)

assert av.__version__ >= "0.4.3", "pyav is out-of-date, please update"


class FileSeekError(Exception):
    pass


class Frame:
    """docstring of Frame"""

    def __init__(self, timestamp, av_frame, index):
        self.timestamp = float(timestamp)
        self.index = int(index)
        self.width = av_frame.width
        self.height = av_frame.height
        self.jpeg_buffer = None
        self.yuv_buffer = None
        self._av_frame = av_frame
        self._img = None
        self._gray = None
        self.is_fake = False

    def copy(self):
        return Frame(self.timestamp, self._av_frame, self.index)

    @property
    def img(self):
        if self._img is None:
            self._img = self._av_frame.to_nd_array(format="bgr24")
        return self._img

    @property
    def bgr(self):
        return self.img

    @property
    def gray(self):
        if self._gray is None:
            plane = self._av_frame.planes[0]
            self._gray = np.frombuffer(plane, np.uint8)
            try:
                self._gray.shape = self.height, self.width
            except ValueError:
                self._gray = self._gray.reshape(-1, plane.line_size)
                self._gray = np.ascontiguousarray(self._gray[:, : self.width])
        return self._gray


class FakeFrame:
    """
    Show FakeFrame when the video is broken or there is a gap between timestamp.
    """

    def __init__(self, width, height, timestamp, index):
        self.timestamp = float(timestamp)
        self.index = int(index)
        self.width = width
        self.height = height
        self.jpeg_buffer = None
        self.yuv_buffer = None
        shape = (self.height, self.width, 3)
        static_img = np.ones(shape, dtype=np.uint8) * 128
        self.img = self.bgr = static_img
        self.is_fake = True

    def copy(self):
        return FakeFrame(self.width, self.height, self.timestamp, self.index)

    @property
    def gray(self):
        return self.img[:, :, 0]  # return first channel


class Decoder(ABC):
    """
    Abstract base class for stream decoders.
    """

    @abstractmethod
    def seek(self, pts_position: int):
        """
        Seek stream decoder to given position (in pts!!)
        """
        pass

    @abstractmethod
    def get_frame_iterator(self) -> T.Iterator[Frame]:
        """
        Returns a fresh iterator for frames, starting from current seek position.
        """
        pass

    @property
    def frame_size(self) -> T.Tuple[int, int]:
        """Frame size in (width, height)"""
        return (
            int(self.video_stream.format.width),
            int(self.video_stream.format.height),
        )

    def cleanup(self):
        """
        Implement for potential cleanup operations on stream close.

        Should be called on stream close.
        """
        pass


class BrokenStream(Decoder):
    @property
    def frame_size(self):
        # fixed frame size
        DEFAULT_WIDTH = 1280
        DEFAULT_HIGHT = 720
        return (DEFAULT_WIDTH, DEFAULT_HIGHT)

    def seek(self, pts_position):
        pass

    def get_frame_iterator(self):
        # returns empty iterator
        yield from ()


class BufferedDecoder(Decoder):
    def __init__(self, container, video_stream):
        self.container = container
        self.video_stream = video_stream
        self._buffered_decoder = self.container.get_buffered_decoder(
            self.video_stream, dec_batch=50, dec_buffer_size=200
        )

    def seek(self, pts_position):
        self._buffered_decoder.seek(pts_position)

    def get_frame_iterator(self):
        return self._buffered_decoder.get_frame()

    def cleanup(self):
        self._buffered_decoder.stop_buffer_thread()


class OnDemandDecoder(Decoder):
    def __init__(self, container, video_stream):
        self.container = container
        self.video_stream = video_stream

    def seek(self, pts_position):
        self.video_stream.seek(pts_position)

    def get_frame_iterator(self):
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                if frame:
                    yield frame


class File_Source(Playback_Source, Base_Source):
    """Simple file capture.

    Playback_Source arguments:
        timing (str): "external", "own" (default), None

    File_Source arguments:
        source_path (str): Path to source file
        loop (bool): loop video set if timing!="external"
        buffered_decoding (bool): use buffered decode
        fill_gaps (bool): fill gaps with static frames
    """

    def __init__(
        self,
        g_pool,
        source_path=None,
        loop=False,
        buffered_decoding=False,
        fill_gaps=False,
        *args,
        **kwargs,
    ):
        super().__init__(g_pool, *args, **kwargs)
        if self.timing == "external":
            self.recent_events = self.recent_events_external_timing
        else:
            self.recent_events = self.recent_events_own_timing
        # minimal attribute set
        self.source_path = str(source_path)
        self.loop = loop
        self.fill_gaps = fill_gaps
        rec, set_name = self.get_rec_set_name(self.source_path)

        self._init_videoset()

        self.timestamps = self.videoset.lookup.timestamp
        if len(self.timestamps) > 1:
            self._frame_rate = (self.timestamps[-1] - self.timestamps[0]) / len(
                self.timestamps
            )
        else:
            # TODO: where does the fallback framerate of 1/20 come from?
            self._frame_rate = 20
        self.buffering = buffered_decoding
        # Load video split for first frame
        self.reset_video()
        self._intrinsics = load_intrinsics(rec, set_name, self.frame_size)

    def get_rec_set_name(self, source_path):
        """
        Return dir and set name by source_path
        """
        rec, file_ = os.path.split(source_path)
        set_name = os.path.splitext(file_)[0]
        return rec, set_name

    def _init_videoset(self):
        rec, set_name = self.get_rec_set_name(self.source_path)
        self.videoset = VideoSet(rec, set_name, self.fill_gaps)
        self.videoset.load_or_build_lookup()
        if self.videoset.is_empty() and self.fill_gaps:
            # create artificial lookup table here

            recording = PupilRecording(rec)
            start_time = recording.meta_info.start_time_synced_s
            if (
                recording.meta_info.recording_software_name
                == recording.meta_info.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # TODO: Currently PI timestamp data is shifted to start at 0 (by
                # subtracting start_time_synced) in order to deal with opengl rounding
                # issues for large numbers in float32 precision. This will change in the
                # future, which will require this to be updated again.
                start_time = 0

            duration = recording.meta_info.duration_s
            # since the eye recordings might be slightly longer than the world recording
            # (due to notification delays) we want to make sure that we generate enough
            # fake world frames to display all eye data, so we make the world recording
            # artificially longer
            BACK_BUFFER_SECONDS = 3
            end_time = start_time + duration + BACK_BUFFER_SECONDS

            fallback_framerate = 30
            timestamps = np.arange(start_time, end_time, 1 / fallback_framerate)
            self.videoset.build_lookup(timestamps)
            assert not self.videoset.is_empty()

    def _setup_video(self, container_index):
        """Setup streams for a given container_index."""
        try:
            self.video_stream.cleanup()
        except AttributeError:
            pass

        if container_index < 0:
            # setup a 'valid' broken stream
            self.video_stream = BrokenStream()
        else:
            container = self.videoset.containers[container_index]
            if container is None:
                # TODO: Shouldn't this be caught through an invalid container_index?
                logger.warning("Video container is broken, although it appeared valid.")
                self.video_stream = BrokenStream()
            else:
                self.video_stream = self._get_streams(container, self.buffering)

        self.video_stream.seek(0)
        self.current_container_index = container_index
        self.frame_iterator = self.video_stream.get_frame_iterator()

    def _get_streams(self, container, should_buffer):
        """Get Video stream from containers."""
        try:
            # look for the first videostream
            video_stream = next(s for s in container.streams if s.type == "video")
        except StopIteration:
            logger.error("Could not find any video stream in the given source file.")
            # fallback to 'valid' broken stream
            return BrokenStream()

        logger.debug(f"loaded videostream: {str(video_stream)}")
        video_stream.thread_count = cpu_count()

        if should_buffer:
            return BufferedDecoder(container, video_stream)
        else:
            return OnDemandDecoder(container, video_stream)

    @property
    def initialised(self):
        return not self.videoset.is_empty()

    @property
    def frame_size(self):
        return self.video_stream.frame_size

    @property
    def frame_rate(self):
        return self._frame_rate

    def get_init_dict(self):
        if self.g_pool.app == "capture":
            settings = super().get_init_dict()
            settings["source_path"] = self.source_path
            settings["loop"] = self.loop
            return settings
        else:
            raise NotImplementedError()

    @property
    def name(self):
        if self.source_path:
            return os.path.splitext(self.source_path)[0]
        else:
            return "File source in ghost mode"

    def get_frame_index(self):
        return int(self.current_frame_idx)

    def get_frame_count(self):
        return int(self.videoset.lookup.size)

    def get_frame(self):
        try:
            target_entry = self.videoset.lookup[self.target_frame_idx]
        except IndexError:
            raise EndofVideoError

        if target_entry.container_idx == -1:
            return self._get_fake_frame_and_advance(target_entry)

        if target_entry.container_idx != self.current_container_index:
            # Contained index changed, need to load other video split
            self._setup_video(target_entry.container_idx)

        # advance frame iterator until we hit the target frame
        for av_frame in self.frame_iterator:
            if not av_frame:
                raise EndofVideoError
            if av_frame.pts == target_entry.pts:
                break
            elif av_frame.pts < target_entry.pts:
                pass
            else:
                # This should never happen, but just in case we should make sure
                # that our current_frame_idx is actually correct afterwards!
                logger.warn("Advancing frame iterator went past the target frame!")
                current_video_lookup = self.videoset.lookup[
                    self.videoset.lookup.container_idx == target_entry.container_idx
                ]
                pts_indices = np.flatnonzero(current_video_lookup.pts == av_frame.pts)
                if pts_indices.size > 1:
                    logger.err("Found multiple maching pts! Something is wrong!")
                    raise EndofVideoError
                elif pts_indices.size == 0:
                    logger.err("Found no maching pts! Something is wrong!")
                    raise EndofVideoError
                self.target_frame_idx = pts_indices[0]
                break

        # update indices, we know that we advanced until target_frame_index!
        self.current_frame_idx = self.target_frame_idx
        self.target_frame_idx += 1
        return Frame(
            timestamp=target_entry.timestamp,
            av_frame=av_frame,
            index=self.current_frame_idx,
        )

    def _get_fake_frame_and_advance(self, target_entry):
        self.current_frame_idx = self.target_frame_idx
        self.target_frame_idx += 1
        return FakeFrame(
            width=self.frame_size[0],
            height=self.frame_size[1],
            timestamp=target_entry.timestamp,
            index=self.current_frame_idx,
        )

    def recent_events_external_timing(self, events):
        try:
            last_index = self._recent_frame.index
        except AttributeError:
            # Get a frame at beginnning
            last_index = -1
        # Seek Frame
        frame = None
        pbt = self.g_pool.seek_control.current_playback_time
        ts_idx = self.g_pool.seek_control.ts_idx_from_playback_time(pbt)
        if ts_idx == last_index:
            frame = self._recent_frame.copy()
        elif ts_idx < last_index or ts_idx > last_index + 1:
            self.seek_to_frame(ts_idx)

        # Normal Case to get next frame
        try:
            frame = frame or self.get_frame()
        except EndofVideoError:
            logger.info("No more video found")
            self.g_pool.seek_control.play = False
            frame = self._recent_frame.copy()
        self.g_pool.seek_control.end_of_seek()
        events["frame"] = frame
        self._recent_frame = frame

    def recent_events_own_timing(self, events):
        if not self.play:
            if self.timing == "own":
                # this is to ensure we don't do full-CPU loops on broken videos when
                # streaming a recording into capture (!)
                sleep(0.05)
            return
        try:
            frame = self.get_frame()
        except EndofVideoError:
            if self.timing == "own":
                # this is to ensure we don't do full-CPU loops on broken videos when
                # streaming a recording into capture (!)
                sleep(0.05)
            logger.info("Video has ended.")
            if self.loop:
                logger.info("Looping enabled. Seeking to beginning.")
                self.reset_video()
                return
            self.notify_all(
                {
                    "subject": "file_source.video_finished",
                    "source_path": self.source_path,
                }
            )
            self.play = False
        else:
            if self.timing == "own":
                self.wait(frame.timestamp)
            self._recent_frame = frame
            events["frame"] = frame

    def seek_to_frame(self, seek_pos):
        try:
            target_entry = self.videoset.lookup[seek_pos]
        except IndexError:
            logger.warning("Seeking to invalid position!")
            return
        if target_entry.container_idx > -1:
            if target_entry.container_idx != self.current_container_index:
                self._setup_video(target_entry.container_idx)
            try:
                # explicit conversion to python int required, else:
                # TypeError: ('Container.seek only accepts integer offset.')
                self.video_stream.seek(int(target_entry.pts))
            except av.AVError as e:
                raise FileSeekError() from e
        else:
            # TODO: Why seek here? Might be inefficient.
            self.video_stream.seek(0)
        # need to re-initialize frame_iterator at the new seek position
        self.frame_iterator = self.video_stream.get_frame_iterator()
        self.finished_sleep = 0
        self.target_frame_idx = seek_pos

    def on_notify(self, notification):
        if (
            notification["subject"] == "file_source.seek"
            and notification.get("source_path") == self.source_path
        ):
            self.seek_to_frame(notification["frame_index"])
        elif (
            notification["subject"] == "file_source.should_play"
            and notification.get("source_path") == self.source_path
        ):
            self.play = True
        elif (
            notification["subject"] == "file_source.should_pause"
            and notification.get("source_path") == self.source_path
        ):
            self.play = False

    def init_ui(self):
        self.add_menu()
        self.menu.label = "File Source: {}".format(os.path.split(self.source_path)[-1])
        from pyglui import ui

        self.menu.append(
            ui.Info_Text(
                "The file source plugin loads and "
                + "displays video from a given file."
            )
        )

        if self.g_pool.app == "capture":

            def toggle_looping(val):
                self.loop = val
                if val:
                    self.play = True

            self.menu.append(ui.Switch("loop", self, setter=toggle_looping))

        self.menu.append(
            ui.Text_Input("source_path", self, label="Full path", setter=lambda x: None)
        )

        self.menu.append(
            ui.Text_Input(
                "frame_size",
                label="Frame size",
                setter=lambda x: None,
                getter=lambda: "{} x {}".format(*self.frame_size),
            )
        )

        self.menu.append(
            ui.Text_Input(
                "frame_rate",
                label="Frame rate",
                setter=lambda x: None,
                getter=lambda: "{:.0f} FPS".format(self.frame_rate),
            )
        )

        self.menu.append(
            ui.Text_Input(
                "frame_num",
                label="Number of frames",
                setter=lambda x: None,
                getter=lambda: self.get_frame_count(),
            )
        )

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        try:
            self.video_stream.cleanup()
        except AttributeError:
            pass
        super().cleanup()

    @property
    def jpeg_support(self):
        return False

    def reset_video(self):
        """
        Initializes video playback to first frame.
        """
        self.current_frame_idx = 0
        self.target_frame_idx = 0
        if not self.initialised:
            self._setup_video(-1)
        else:
            container_idx_of_first_frame = self.videoset.lookup[0].container_idx
            self._setup_video(container_idx_of_first_frame)


class File_Manager(Base_Manager):
    """Summary

    Attributes:
        file_exts (list): File extensions to filter displayed files
        root_folder (str): Folder path, which includes file sources
    """

    gui_name = "Video File Source"
    file_exts = [".mp4", ".mkv", ".mov", ".mjpeg"]

    def __init__(self, g_pool, root_folder=None):
        super().__init__(g_pool)
        base_dir = self.g_pool.user_dir.rsplit(os.path.sep, 1)[0]
        default_rec_dir = os.path.join(base_dir, "recordings")
        self.root_folder = root_folder or default_rec_dir

    def init_ui(self):
        self.add_menu()
        from pyglui import ui

        self.add_auto_select_button()
        self.menu.append(
            ui.Info_Text(
                "Enter a folder to enumerate all eligible video files. "
                + "Be aware that entering folders with a lot of files can "
                + "slow down Pupil Capture."
            )
        )

        def set_root(folder):
            if not os.path.isdir(folder):
                logger.error("`%s` is not a valid folder path." % folder)
            else:
                self.root_folder = folder

        self.menu.append(
            ui.Text_Input("root_folder", self, label="Source Folder", setter=set_root)
        )

        def split_enumeration():
            eligible_files = self.enumerate_folder(self.root_folder)
            eligible_files.insert(0, (None, "Select to activate"))
            return zip(*eligible_files)

        self.menu.append(
            ui.Selector(
                "selected_file",
                selection_getter=split_enumeration,
                getter=lambda: None,
                setter=self.activate,
                label="Video File",
            )
        )

    def deinit_ui(self):
        self.remove_menu()

    def activate(self, full_path):
        if not full_path:
            return
        settings = {"source_path": full_path, "timing": "own"}
        self.activate_source(settings)

    def auto_activate_source(self):
        self.activate(None)

    def on_drop(self, paths):
        for p in paths:
            if os.path.splitext(p)[-1] in self.file_exts:
                self.activate(p)
                return True
        return False

    def enumerate_folder(self, path):
        eligible_files = []
        is_eligible = lambda f: os.path.splitext(f)[-1] in self.file_exts
        path = os.path.abspath(os.path.expanduser(path))
        for root, dirs, files in os.walk(path):

            def root_split(file):
                full_p = os.path.join(root, file)
                disp_p = full_p.replace(path, "")
                return (full_p, disp_p)

            eligible_files.extend(map(root_split, filter(is_eligible, files)))
        eligible_files.sort(key=lambda x: x[1])
        return eligible_files

    def get_init_dict(self):
        return {"root_folder": self.root_folder}

    def activate_source(self, settings={}):
        if self.g_pool.process == "world":
            self.notify_all(
                {"subject": "start_plugin", "name": "File_Source", "args": settings}
            )
        else:
            self.notify_all(
                {
                    "subject": "start_eye_plugin",
                    "target": self.g_pool.process,
                    "name": "File_Source",
                    "args": settings,
                }
            )

    def recent_events(self, events):
        pass
