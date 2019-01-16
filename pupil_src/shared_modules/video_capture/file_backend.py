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
import sys
import numpy as np

from multiprocessing import cpu_count
from time import sleep
from camera_models import load_intrinsics
from .utils import VideoSet

from .base_backend import Base_Manager, Base_Source, EndofVideoError, Playback_Source

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
av.logging.set_level(av.logging.ERROR)
logging.getLogger("libav").setLevel(logging.ERROR)

assert av.__version__ >= "0.4.2", "pyav is out-of-date, please update"
assert sys.version_info >= (3, 7), "We need python 3.7 or later to run Pupil player"


class FileSeekError(Exception):
    pass


class Frame(object):
    """docstring of Frame"""

    def __init__(self, timestamp, av_frame, index):
        self._av_frame = av_frame
        self.timestamp = timestamp
        self.index = index
        self._img = None
        self._gray = None
        self.jpeg_buffer = None
        self.yuv_buffer = None
        self.height, self.width = av_frame.height, av_frame.width

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
    static_img = np.ones((720, 1280), dtype=np.uint8)

    def __init__(self, timestamp, index):
        self.yuv_buffer = None
        self.img = self.bgr = self.static_img.copy()
        self.timestamp = timestamp
        self.index = index


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
        self._initialised = True
        self.source_path = source_path
        self.timestamps = None
        self.frame_count = []
        self.loop = loop
        self.buffering = buffered_decoding
        self.fill_gaps = fill_gaps

        if not source_path or not os.path.isfile(source_path):
            logger.error(
                "Init failed. Source file could not be found at `%s`" % source_path
            )
            self._initialised = False
            return
        rec, file_ = os.path.split(source_path)
        set_name = os.path.splitext(file_)[0]
        self.videoset = VideoSet(rec, set_name)
        try:
            self.videoset.load_lookup()
        except FileNotFoundError:
            self.videoset.build_lookup()

        if not self.fill_gaps:
            # filter filled lookup entries
            cont_idc = self.videoset.lookup.container_idx
            self.videoset.lookup = self.videoset.lookup[cont_idc > -1]

        self.timestamps = self.videoset.lookup.timestamp
        self.current_container_index = self.videoset.lookup.container_idx[0]
        self.target_frame_idx = 0
        self.current_frame_idx = 0

        self.setup_video(self.current_container_index)  # load first split

        self._intrinsics = load_intrinsics(rec, set_name, self.frame_size)

    def setup_video(self, container_index):
        self.current_container_index = container_index
        self.container = self.videoset.videos_container[container_index]
        self.video_stream, self.audio_stream = self._get_streams(self.container)

        # set the pts rate to convert pts to frame index.
        # We use videos with pts writte like indecies.
        if self.buffering:
            self.buffered_decoder = self.container.get_buffered_decoder(
                self.video_stream, dec_batch=50, dec_buffer_size=200
            )
            self.next_frame = self.buffered_decoder.get_frame()
        else:
            self.next_frame = self._next_frame()

        f0, f1 = next(self.next_frame), next(self.next_frame)
        self.pts_rate = f1.pts
        self.video_stream.seek(f0.pts)
        self.average_rate = (self.timestamps[-1] - self.timestamps[0]) / len(
            self.timestamps
        )

    def ensure_initialisation(fallback_func=None, requires_playback=False):
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def run_func(self, *args, **kwargs):
                if self._initialised and self.video_stream:
                    # test self.play only if requires_playback is True
                    if not requires_playback or self.play:
                        return func(self, *args, **kwargs)
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    logger.debug("Initialisation required.")

            return run_func

        return decorator

    def _get_streams(self, container):
        """
        Get Video and Audio stream from containers
        """
        try:
            video_stream = next(
                s for s in container.streams if s.type == "video"
            )  # looking for the first videostream
            logger.debug("loaded videostream: %s" % video_stream)
            video_stream.thread_count = cpu_count()
        except StopIteration:
            video_stream = None
            logger.error("No videostream found in media container")

        try:
            audio_stream = next(
                s for s in container.streams if s.type == "audio"
            )  # looking for the first audiostream
            logger.debug("loaded audiostream: %s" % audio_stream)
        except StopIteration:
            audio_stream = None
            logger.debug("No audiostream found in media container")
        if not video_stream and not audio_stream:
            logger.error(
                "Init failed. Could not find any video or audio"
                + "stream in the given source file."
            )
            self._initialised = False
            return
        return video_stream, audio_stream

    @property
    def initialised(self):
        return self._initialised

    @property
    @ensure_initialisation(fallback_func=lambda: (640, 480))
    def frame_size(self):
        return (
            int(self.video_stream.format.width),
            int(self.video_stream.format.height),
        )

    @property
    @ensure_initialisation(fallback_func=lambda: 20)
    def frame_rate(self):
        return 1.0 / float(self.average_rate)

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
        return self.current_frame_idx

    def get_frame_count(self):
        return self.videoset.lookup.size

    @ensure_initialisation()
    def _next_frame(self):
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                if frame:
                    yield frame

    def _convert_frame_index(self, pts):
        """
        Calculate frame index by current_container_index
        """
        cont_frame_idx = self.pts_to_idx(pts)
        cont_mask = np.isin(
            self.videoset.lookup.container_idx, self.current_container_index
        )
        frame_mask = np.isin(self.videoset.lookup.container_frame_idx, cont_frame_idx)
        videoset_idx = np.flatnonzero(cont_mask & frame_mask)[0]
        return videoset_idx

    @ensure_initialisation()
    def pts_to_idx(self, pts):
        # some older mkv did not use perfect timestamping
        # so we are doing int(round()) to clear that.
        # With properly spaced pts (any v0.6.100+ recording)
        # just int() would suffice.
        return int(pts / self.pts_rate)

    @ensure_initialisation()
    def idx_to_pts(self, idx):
        return idx * self.pts_rate

    @ensure_initialisation()
    def get_frame(self):
        while 1:
            target_entry = self.videoset.lookup[self.target_frame_idx]
            if target_entry.container_idx == -1:
                self.current_frame_idx = self.target_frame_idx
                self.target_frame_idx += 1
                return FakeFrame(target_entry.timestamp, self.target_frame_idx)
            elif target_entry.container_idx != self.current_container_index:
                self.setup_video(target_entry.container_idx)
            try:
                av_frame = next(self.next_frame)
                index = self._convert_frame_index(av_frame.pts)
                if index == self.target_frame_idx:
                    break
                elif index < self.target_frame_idx:
                    pass
            except StopIteration as stop:
                raise EndofVideoError from stop

            # if self.loop:
            #     logger.info("Looping enabled. Seeking to beginning.")
            #     self.seek_to_frame(0)
            #     return self.get_frame()
            # else:
            #     logger.debug(
            #         "End of videofile %s %s"
            #         % (self.current_frame_idx, len(self.timestamps))
            #     )
            #     raise EndofVideoError("Reached end of video file")

        self.target_frame_idx = index + 1
        self.current_frame_idx = index
        return Frame(target_entry.timestamp, av_frame, index=index)

    @ensure_initialisation(fallback_func=lambda evt: sleep(0.05))
    def recent_events_external_timing(self, events):
        try:
            last_index = self._recent_frame.index
        except AttributeError:
            # Get a frame at beginnning
            frame = self.get_frame()
            self._recent_frame = frame
            last_index = -1
        # Seek Frame
        pbt = self.g_pool.seek_control.current_playback_time
        ts_idx = self.g_pool.seek_control.ts_idx_from_playback_time(pbt)
        if ts_idx < last_index or ts_idx > last_index + 1:
            self.seek_to_frame(ts_idx)

        # Normal Case to get next frame
        if not self.play:
            frame = self._recent_frame.copy()
        else:
            try:
                frame = self.get_frame()
            except EndofVideoError:
                logger.info("No more video found")
                self.g_pool.seek_control.play = False
                frame = self._recent_frame.copy()
        self.g_pool.seek_control.end_of_seek()
        events["frame"] = frame
        self._recent_frame = frame

    @ensure_initialisation(
        fallback_func=lambda evt: sleep(0.05), requires_playback=True
    )
    def recent_events_own_timing(self, events):
        try:
            frame = self.get_frame()
        except EndofVideoError:
            logger.info("Video has ended.")
            self.notify_all(
                {
                    "subject": "file_source.video_finished",
                    "source_path": self.source_path,
                }
            )
            self.play = False
        else:
            if self.timing:
                self.wait(frame.timestamp)
            self._recent_frame = frame
            events["frame"] = frame

    @ensure_initialisation()
    def seek_to_frame(self, seek_pos):
        target_entry = self.videoset.lookup[seek_pos]
        if target_entry.container_idx > -1:
            if target_entry.container_idx != self.current_container_index:
                self.setup_video(target_entry.container_idx)
            try:
                # explicit conversion to python int required, else:
                # TypeError: ('Container.seek only accepts integer offset.',
                target_pts = int(self.idx_to_pts(target_entry.container_frame_idx))
                if self.buffering:
                    self.buffered_decoder.seek(target_pts)
                else:
                    self.video_stream.seek(target_pts)
            except av.AVError as e:
                raise FileSeekError() from e
            else:
                if not self.buffering:
                    self.next_frame = self._next_frame()
                else:
                    raise NotImplementedError
        self.finished_sleep = 0
        self.target_frame_idx = seek_pos
        self.play = True

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

    def seek_to_prev_frame(self):
        self.seek_to_frame(max(0, self.current_frame_idx - 1))

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
            self.buffered_decoder.stop_buffer_thread()
        except AttributeError:
            pass
        super().cleanup()

    @property
    def jpeg_support(self):
        return False


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

    def on_drop(self, paths):
        for p in paths:
            if os.path.splitext(p)[-1] in self.file_exts:
                self.activate(p)
                return

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
                    "subject": "start_eye_capture",
                    "target": self.g_pool.process,
                    "name": "File_Source",
                    "args": settings,
                }
            )

    def recent_events(self, events):
        pass
