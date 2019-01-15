"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import time
import cv2
import os

import pyrealsense2 as rs

from version_utils import VersionFormat
from .base_backend import Base_Source, Base_Manager
from av_writer import AV_Writer
from camera_models import load_intrinsics

import glfw
import gl_utils
from OpenGL.GL import *
from OpenGL.GLU import *
from pyglui import cygl
import cython_methods
import numpy as np
from ctypes import *

# check versions for our own depedencies as they are fast-changing
# assert VersionFormat(rs.__version__) >= VersionFormat("2.2") # FIXME

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


TIMEOUT = 500  # ms FIXME
DEFAULT_COLOR_SIZE = (1280, 720)
DEFAULT_COLOR_FPS = 30
DEFAULT_DEPTH_SIZE = (640, 480)
DEFAULT_DEPTH_FPS = 30

# very thin wrapper for rs.frame objects
class ColorFrame(object):
    def __init__(self, data, timestamp, index):
        self.timestamp = timestamp
        self.index = index

        self.data = data[:, :, np.newaxis].view(dtype=np.uint8)
        total_size = self.data.size
        y_plane = total_size // 2
        u_plane = y_plane // 2
        self._yuv = np.empty(total_size, dtype=np.uint8)
        self._yuv[:y_plane] = self.data[:, :, 0].ravel()
        self._yuv[y_plane : y_plane + u_plane] = self.data[:, ::2, 1].ravel()
        self._yuv[y_plane + u_plane :] = self.data[:, 1::2, 1].ravel()
        self._shape = self.data.shape[:2]

        self._bgr = None
        self._gray = None

    @property
    def height(self):
        return self._shape[0]

    @property
    def width(self):
        return self._shape[1]

    @property
    def yuv_buffer(self):
        return self._yuv

    @property
    def yuv422(self):
        Y = self._yuv[: self._yuv.size // 2]
        U = self._yuv[self._yuv.size // 2 : 3 * self._yuv.size // 4]
        V = self._yuv[3 * self._yuv.size // 4 :]

        Y.shape = self._shape
        U.shape = self._shape[0], self._shape[1] // 2
        V.shape = self._shape[0], self._shape[1] // 2

        return Y, U, V

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = cv2.cvtColor(self.data, cv2.COLOR_YUV2BGR_YUYV)
        return self._bgr

    @property
    def img(self):
        return self.bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self._yuv[: self._yuv.size // 2]
            self._gray.shape = self._shape
        return self._gray


class DepthFrame(object):
    def __init__(self, data, timestamp, index):
        self.timestamp = timestamp
        self.index = index

        self._bgr = None
        self._gray = None
        self.depth = data
        self.yuv_buffer = None

    @property
    def height(self):
        return self.depth.shape[0]

    @property
    def width(self):
        return self.depth.shape[1]

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = cython_methods.cumhist_color_map16(self.depth)
        return self._bgr

    @property
    def img(self):
        return self.bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.bgr, cv2.cv2.COLOR_BGR2GRAY)
        return self._gray


class Realsense2_Source(Base_Source):
    def __init__(
        self,
        g_pool,
        device_id=None,
        frame_size=DEFAULT_COLOR_SIZE,
        frame_rate=DEFAULT_COLOR_FPS,
        depth_frame_size=DEFAULT_DEPTH_SIZE,
        depth_frame_rate=DEFAULT_DEPTH_FPS,
        preview_depth=False,
        device_options=(),
        record_depth=True,
    ):
        logger.debug("_init_ started")
        super().__init__(g_pool)
        self._intrinsics = None
        self.color_frame_index = 0
        self.depth_frame_index = 0
        self.context = rs.context()
        self.pipeline = rs.pipeline(self.context)
        self.pipeline_profile = None
        self.preview_depth = preview_depth
        self.record_depth = record_depth
        self.depth_video_writer = None
        self._needs_restart = False
        self.frame_size_backup = DEFAULT_COLOR_SIZE
        self.depth_frame_size_backup = DEFAULT_DEPTH_SIZE
        self.frame_rate_backup = DEFAULT_COLOR_FPS
        self.depth_frame_rate_backup = DEFAULT_DEPTH_FPS

        self._initialize_device(
            device_id,
            frame_size,
            frame_rate,
            depth_frame_size,
            depth_frame_rate,
            device_options,
        )
        logger.debug("_init_ completed")

    def _initialize_device(
        self,
        device_id,
        color_frame_size,
        color_fps,
        depth_frame_size,
        depth_fps,
        device_options=(),
    ):
        self.stop_pipeline()
        self.last_color_frame_ts = None
        self.last_depth_frame_ts = None
        self._recent_frame = None
        self._recent_depth_frame = None

        if device_id is None:
            device_id = self.device_id

        if device_id is None:  # FIXME these two if blocks look ugly.
            return

        # use default streams to filter modes by rs_stream and rs_format
        self._available_modes = self._enumerate_formats(device_id)
        logger.debug(
            "device_id: {} self._available_modes: {}".format(
                device_id, str(self._available_modes)
            )
        )

        if (
            color_frame_size is not None
            and depth_frame_size is not None
            and color_fps is not None
            and depth_fps is not None
        ):
            color_frame_size = tuple(color_frame_size)
            depth_frame_size = tuple(depth_frame_size)

            logger.debug(
                "Initialize with Color {}@{}\tDepth {}@{}".format(
                    color_frame_size, color_fps, depth_frame_size, depth_fps
                )
            )

            # make sure the frame rates are compatible with the given frame sizes
            color_fps = self._get_valid_frame_rate(
                rs.stream.color, color_frame_size, color_fps
            )
            depth_fps = self._get_valid_frame_rate(
                rs.stream.depth, depth_frame_size, depth_fps
            )

            self.frame_size_backup = color_frame_size
            self.depth_frame_size_backup = depth_frame_size
            self.frame_rate_backup = color_fps
            self.depth_frame_rate_backup = depth_fps

            config = self._prep_configuration(
                color_frame_size, color_fps, depth_frame_size, depth_fps
            )
        else:
            config = self._get_default_config()
            self.frame_size_backup = DEFAULT_COLOR_SIZE
            self.depth_frame_size_backup = DEFAULT_DEPTH_SIZE
            self.frame_rate_backup = DEFAULT_COLOR_FPS
            self.depth_frame_rate_backup = DEFAULT_DEPTH_FPS

        try:
            self.pipeline_profile = self.pipeline.start(config)
        except RuntimeError as re:
            logger.error("Cannot start pipeline! " + str(re))
            self.pipeline_profile = None
        else:
            self.stream_profiles = {
                s.stream_type(): s.as_video_stream_profile()
                for s in self.pipeline_profile.get_streams()
            }
            logger.debug("Pipeline started for device " + device_id)
            logger.debug("Stream profiles: " + str(self.stream_profiles))

            self._intrinsics = load_intrinsics(
                self.g_pool.user_dir, self.name, self.frame_size
            )
            self.update_menu()
            self._needs_restart = False

    def _prep_configuration(
        self,
        color_frame_size=None,
        color_fps=None,
        depth_frame_size=None,
        depth_fps=None,
    ):
        config = rs.config()

        # only use these two formats
        color_format = rs.format.yuyv
        depth_format = rs.format.z16

        config.enable_stream(
            rs.stream.depth,
            depth_frame_size[0],
            depth_frame_size[1],
            depth_format,
            depth_fps,
        )

        config.enable_stream(
            rs.stream.color,
            color_frame_size[0],
            color_frame_size[1],
            color_format,
            color_fps,
        )

        return config

    def _get_default_config(self):
        config = rs.config()  # default config is RGB8, we want YUYV
        config.enable_stream(
            rs.stream.color,
            DEFAULT_COLOR_SIZE[0],
            DEFAULT_COLOR_SIZE[1],
            rs.format.yuyv,
            DEFAULT_COLOR_FPS,
        )
        config.enable_stream(
            rs.stream.depth,
            DEFAULT_DEPTH_SIZE[0],
            DEFAULT_DEPTH_SIZE[1],
            rs.format.z16,
            DEFAULT_DEPTH_FPS,
        )
        return config

    def _get_valid_frame_rate(self, stream_type, frame_size, fps):
        assert stream_type == rs.stream.color or stream_type == rs.stream.depth

        if not self._available_modes or stream_type not in self._available_modes:
            logger.warning(
                "_get_valid_frame_rate: self._available_modes not set yet. Returning default fps."
            )
            if stream_type == rs.stream.color:
                return DEFAULT_COLOR_FPS
            elif stream_type == rs.stream.depth:
                return DEFAULT_DEPTH_FPS
            else:
                raise ValueError("Unexpected `stream_type`: {}".format(stream_type))

        if frame_size not in self._available_modes[stream_type]:
            logger.error(
                "Frame size not supported for {}: {}. Returning default fps".format(
                    stream_type, frame_size
                )
            )
            if stream_type == rs.stream.color:
                return DEFAULT_COLOR_FPS
            elif stream_type == rs.stream.depth:
                return DEFAULT_DEPTH_FPS

        if fps not in self._available_modes[stream_type][frame_size]:
            old_fps = fps
            rates = [
                abs(r - fps) for r in self._available_modes[stream_type][frame_size]
            ]
            best_rate_idx = rates.index(min(rates))
            fps = self._available_modes[stream_type][frame_size][best_rate_idx]
            logger.warning(
                "{} fps is not supported for ({}) for Color Stream. Fallback to {} fps".format(
                    old_fps, frame_size, fps
                )
            )

        return fps

    def _enumerate_formats(self, device_id):
        """Enumerate formats into hierachical structure:

        streams:
            resolutions:
                framerates
        """
        formats = {}

        if self.context is None:
            return formats

        devices = self.context.query_devices()
        current_device = None

        for d in devices:
            try:
                serial = d.get_info(rs.camera_info.serial_number)
            except RuntimeError as re:
                logger.error("Device no longer available " + str(re))
            else:
                if device_id == serial:
                    current_device = d

        if current_device is None:
            return formats
        logger.debug("Found the current device: " + device_id)

        sensors = current_device.query_sensors()
        for s in sensors:
            stream_profiles = s.get_stream_profiles()
            for sp in stream_profiles:
                vp = sp.as_video_stream_profile()
                stream_type = vp.stream_type()

                if stream_type not in (rs.stream.color, rs.stream.depth):
                    continue
                elif vp.format() not in (rs.format.z16, rs.format.yuyv):
                    continue

                formats.setdefault(stream_type, {})
                stream_resolution = (vp.width(), vp.height())
                formats[stream_type].setdefault(stream_resolution, []).append(vp.fps())

        return formats

    def stop_pipeline(self):
        if self.online:
            try:
                self.pipeline_profile = None
                self.stream_profiles = None
                self.pipeline.stop()
                logger.debug("Pipeline stopped.")
            except RuntimeError as re:
                logger.error("Cannot stop the pipeline: " + str(re))

    def cleanup(self):
        if self.depth_video_writer is not None:
            self.stop_depth_recording()
        self.stop_pipeline()

    def get_init_dict(self):
        return {
            "frame_size": self.frame_size,
            "frame_rate": self.frame_rate,
            "depth_frame_size": self.depth_frame_size,
            "depth_frame_rate": self.depth_frame_rate,
            "preview_depth": self.preview_depth,
            "record_depth": self.record_depth,
        }

    def get_frames(self):
        if self.online:
            try:
                frames = self.pipeline.wait_for_frames(TIMEOUT)
            except RuntimeError as e:
                logger.error("get_frames: Timeout!")
                raise RuntimeError(e)
            else:
                current_time = self.g_pool.get_timestamp()

                color = None
                # if we're expecting color frames
                if rs.stream.color in self.stream_profiles:
                    color_frame = frames.get_color_frame()
                    last_color_frame_ts = color_frame.get_timestamp()
                    if self.last_color_frame_ts != last_color_frame_ts:
                        self.last_color_frame_ts = last_color_frame_ts
                        color = ColorFrame(
                            np.asanyarray(color_frame.get_data()),
                            current_time,
                            self.color_frame_index,
                        )
                        self.color_frame_index += 1

                depth = None
                # if we're expecting depth frames
                if rs.stream.depth in self.stream_profiles:
                    depth_frame = frames.get_depth_frame()
                    last_depth_frame_ts = depth_frame.get_timestamp()
                    if self.last_depth_frame_ts != last_depth_frame_ts:
                        self.last_depth_frame_ts = last_depth_frame_ts
                        depth = DepthFrame(
                            np.asanyarray(depth_frame.get_data()),
                            current_time,
                            self.depth_frame_index,
                        )
                        self.depth_frame_index += 1

                return color, depth
        return None, None

    def recent_events(self, events):
        if self._needs_restart or not self.online:
            logger.debug("recent_events -> restarting device")
            self.restart_device()
            time.sleep(0.01)
            return

        try:
            color_frame, depth_frame = self.get_frames()
        except RuntimeError as re:
            logger.warning("Realsense failed to provide frames." + str(re))
            self._recent_frame = None
            self._recent_depth_frame = None
            self._needs_restart = True
        else:
            if color_frame is not None:
                self._recent_frame = color_frame
                events["frame"] = color_frame

            if depth_frame is not None:
                self._recent_depth_frame = depth_frame
                events["depth_frame"] = depth_frame

                if self.depth_video_writer is not None:
                    self.depth_video_writer.write_video_frame(depth_frame)

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Local USB Video Source"
        self.update_menu()

    def update_menu(self):
        logger.debug("update_menu")
        try:
            del self.menu[:]
        except AttributeError:
            return

        from pyglui import ui

        if not self.online:
            self.menu.append(ui.Info_Text("Capture initialization failed."))
            return

        self.menu.append(ui.Switch("record_depth", self, label="Record Depth Stream"))
        self.menu.append(ui.Switch("preview_depth", self, label="Preview Depth"))

        if self._available_modes is not None:

            def frame_size_selection_getter():
                if self.device_id:
                    frame_size = sorted(
                        self._available_modes[rs.stream.color], reverse=True
                    )
                    labels = ["({}, {})".format(t[0], t[1]) for t in frame_size]
                    return frame_size, labels
                else:
                    return [self.frame_size_backup], [str(self.frame_size_backup)]

            selector = ui.Selector(
                "frame_size",
                self,
                selection_getter=frame_size_selection_getter,
                label="Color Resolution",
            )
            self.menu.append(selector)

            def frame_rate_selection_getter():
                if self.device_id:
                    avail_fps = [
                        fps
                        for fps in self._available_modes[rs.stream.color][
                            self.frame_size
                        ]
                    ]
                    return avail_fps, [str(fps) for fps in avail_fps]
                else:
                    return [self.frame_rate_backup], [str(self.frame_rate_backup)]

            selector = ui.Selector(
                "frame_rate",
                self,
                selection_getter=frame_rate_selection_getter,
                label="Color Frame Rate",
            )
            self.menu.append(selector)

            def depth_frame_size_selection_getter():
                if self.device_id:
                    depth_sizes = sorted(
                        self._available_modes[rs.stream.depth], reverse=True
                    )
                    labels = ["({}, {})".format(t[0], t[1]) for t in depth_sizes]
                    return depth_sizes, labels
                else:
                    return (
                        [self.depth_frame_size_backup],
                        [str(self.depth_frame_size_backup)],
                    )

            selector = ui.Selector(
                "depth_frame_size",
                self,
                selection_getter=depth_frame_size_selection_getter,
                label="Depth Resolution",
            )
            self.menu.append(selector)

            def depth_frame_rate_selection_getter():
                if self.device_id:
                    avail_fps = [
                        fps
                        for fps in self._available_modes[rs.stream.depth][
                            self.depth_frame_size
                        ]
                    ]
                    return avail_fps, [str(fps) for fps in avail_fps]
                else:
                    return (
                        [self.depth_frame_rate_backup],
                        [str(self.depth_frame_rate_backup)],
                    )

            selector = ui.Selector(
                "depth_frame_rate",
                self,
                selection_getter=depth_frame_rate_selection_getter,
                label="Depth Frame Rate",
            )
            self.menu.append(selector)

            def reset_options():
                logger.debug("reset_options")
                self.reset_device(self.device_id)

            sensor_control = ui.Growing_Menu(label="Sensor Settings")
            sensor_control.append(
                ui.Button("Reset device options to default", reset_options)
            )
            self.menu.append(sensor_control)
        else:
            logger.debug("update_menu: self._available_modes is None")

    def gl_display(self):

        if self.preview_depth and self._recent_depth_frame is not None:
            self.g_pool.image_tex.update_from_ndarray(self._recent_depth_frame.bgr)
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw()
        elif self._recent_frame is not None:
            self.g_pool.image_tex.update_from_yuv_buffer(
                self._recent_frame.yuv_buffer,
                self._recent_frame.width,
                self._recent_frame.height,
            )
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw()

        if not self.online:
            super().gl_display()

        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3)
        )

    def reset_device(self, device_id):
        logger.debug("reset_device")
        if device_id is None:
            device_id = self.device_id

        self.notify_all(
            {
                "subject": "realsense2_source.restart",
                "device_id": device_id,
                "color_frame_size": None,
                "color_fps": None,
                "depth_frame_size": None,
                "depth_fps": None,
                "device_options": [],  # FIXME
            }
        )

    def restart_device(
        self,
        color_frame_size=None,
        color_fps=None,
        depth_frame_size=None,
        depth_fps=None,
        device_options=None,
    ):
        if color_frame_size is None:
            color_frame_size = self.frame_size
        if color_fps is None:
            color_fps = self.frame_rate
        if depth_frame_size is None:
            depth_frame_size = self.depth_frame_size
        if depth_fps is None:
            depth_fps = self.depth_frame_rate
        if device_options is None:
            device_options = []  # FIXME

        self.notify_all(
            {
                "subject": "realsense2_source.restart",
                "device_id": None,
                "color_frame_size": color_frame_size,
                "color_fps": color_fps,
                "depth_frame_size": depth_frame_size,
                "depth_fps": depth_fps,
                "device_options": device_options,
            }
        )
        logger.debug("self.restart_device --> self.notify_all")

    def on_click(self, pos, button, action):
        pass

    def on_notify(self, notification):
        logger.debug(
            'self.on_notify, notification["subject"]: ' + notification["subject"]
        )
        if notification["subject"] == "realsense2_source.restart":
            kwargs = notification.copy()
            del kwargs["subject"]
            del kwargs["topic"]
            self._initialize_device(**kwargs)
        elif notification["subject"] == "recording.started":
            self.start_depth_recording(notification["rec_path"])
        elif notification["subject"] == "recording.stopped":
            self.stop_depth_recording()

    def start_depth_recording(self, rec_loc):
        if not self.record_depth:
            return

        if self.depth_video_writer is not None:
            logger.warning("Depth video recording has been started already")
            return

        video_path = os.path.join(rec_loc, "depth.mp4")
        self.depth_video_writer = AV_Writer(
            video_path, fps=self.depth_frame_rate, use_timestamps=True
        )

    def stop_depth_recording(self):
        if self.depth_video_writer is None:
            logger.warning("Depth video recording was not running")
            return

        self.depth_video_writer.close()
        self.depth_video_writer = None

    @property
    def device_id(self):
        if self.online:  # already running
            return self.pipeline_profile.get_device().get_info(
                rs.camera_info.serial_number
            )
        else:
            # set the first available device
            devices = self.context.query_devices()
            if devices:
                logger.info("device_id: first device by default.")
                return devices[0].get_info(rs.camera_info.serial_number)
            else:
                logger.debug("device_id: No device connected.")
                return None

    @property
    def frame_size(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.color]
            # TODO check width & height is in self.available modes
            return stream_profile.width(), stream_profile.height()
        except AttributeError:
            return self.frame_size_backup
        except KeyError:
            return self.frame_size_backup
        except TypeError:
            return self.frame_size_backup

    @frame_size.setter
    def frame_size(self, new_size):
        if new_size != self.frame_size:
            self.restart_device(color_frame_size=new_size)

    @property
    def frame_rate(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.color]
            # TODO check FPS is in self.available modes
            return stream_profile.fps()
        except AttributeError:
            return self.frame_rate_backup
        except KeyError:
            return self.frame_rate_backup
        except TypeError:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        if new_rate != self.frame_rate:
            self.restart_device(color_fps=new_rate)

    @property
    def depth_frame_size(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.depth]
            # TODO check width & height is in self.available modes
            return stream_profile.width(), stream_profile.height()
        except AttributeError:
            return self.depth_frame_size_backup
        except KeyError:
            return self.depth_frame_size_backup
        except TypeError:
            return self.depth_frame_size_backup

    @depth_frame_size.setter
    def depth_frame_size(self, new_size):
        if new_size != self.depth_frame_size:
            self.restart_device(depth_frame_size=new_size)

    @property
    def depth_frame_rate(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.depth]
            return stream_profile.fps()
        except AttributeError:
            return self.depth_frame_rate_backup
        except KeyError:
            return self.depth_frame_rate_backup
        except TypeError:
            return self.depth_frame_rate_backup

    @depth_frame_rate.setter
    def depth_frame_rate(self, new_rate):
        if new_rate != self.depth_frame_rate:
            self.restart_device(depth_fps=new_rate)

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return self.pipeline_profile is not None and self.pipeline is not None

    @property
    def name(self):
        if self.online:
            return self.pipeline_profile.get_device().get_info(rs.camera_info.name)
        else:
            logger.debug(
                "self.name: Realsense2 not online. Falling back to Ghost capture"
            )
            return "Ghost capture"


class Realsense2_Manager(Base_Manager):
    """Manages Intel RealSense D400 sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """

    gui_name = "RealSense D400"

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()
        from pyglui import ui

        self.menu.append(ui.Info_Text("Intel RealSense D400 sources"))

        def is_streaming(device_id):
            try:
                c = rs.config()
                c.enable_device(device_id)  # device_id is in fact the serial_number
                p = rs.pipeline()
                p.start(c)
                p.stop()
                return False
            except RuntimeError:
                return True

        def get_device_info(d):
            name = d.get_info(rs.camera_info.name)  # FIXME is camera in use?
            device_id = d.get_info(rs.camera_info.serial_number)

            fmt = "- " if is_streaming(device_id) else ""
            fmt += name

            return device_id, fmt

        def dev_selection_list():
            default = (None, "Select to activate")
            try:
                ctx = rs.context()  # FIXME cannot use "with rs.context() as ctx:"
                # got "AttributeError: __enter__"
                # see https://stackoverflow.com/questions/5093382/object-becomes-none-when-using-a-context-manager
                dev_pairs = [default] + [get_device_info(d) for d in ctx.devices]
            except Exception:  # FIXME
                dev_pairs = [default]

            return zip(*dev_pairs)

        def activate(source_uid):
            if source_uid is None:
                return

            settings = {
                "frame_size": self.g_pool.capture.frame_size,
                "frame_rate": self.g_pool.capture.frame_rate,
                "device_id": source_uid,
            }
            if self.g_pool.process == "world":
                self.notify_all(
                    {
                        "subject": "start_plugin",
                        "name": "Realsense2_Source",
                        "args": settings,
                    }
                )
            else:
                self.notify_all(
                    {
                        "subject": "start_eye_capture",
                        "target": self.g_pool.process,
                        "name": "Realsense2_Source",
                        "args": settings,
                    }
                )

        self.menu.append(
            ui.Selector(
                "selected_source",
                selection_getter=dev_selection_list,
                getter=lambda: None,
                setter=activate,
                label="Activate source",
            )
        )

    def deinit_ui(self):
        self.remove_menu()
