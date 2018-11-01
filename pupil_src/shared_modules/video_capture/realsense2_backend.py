"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

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
logging.getLogger("pyrealsense2").setLevel(logging.ERROR + 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# FIXME
TIMEOUT = 1000  # ms

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
        frame_size=(640, 480),
        frame_rate=30,
        depth_frame_size=(640, 480),
        depth_frame_rate=30,
        preview_depth=False,
        device_options=(),
        record_depth=False,
    ):
        logger.debug("_init_ started")
        super().__init__(g_pool)
        self._intrinsics = None
        self.color_frame_index = 0
        self.depth_frame_index = 0
        self.device_id = None  # we'll use serial_number for this
        self.context = rs.context()
        self.pipeline = rs.pipeline(self.context)
        self.pipeline_profile = None
        self.preview_depth = preview_depth
        self.record_depth = record_depth
        self.depth_video_writer = None
        self.controls = {}
        self.pitch = 0
        self.yaw = 0
        self.last_pos = (0, 0)
        self.depth_window = None
        self._needs_restart = False

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
        devices = self.context.query_devices()  # of type pyrealsense2.device_list
        logger.debug("_initialize_device: color_frame_size " + str(color_frame_size))
        logger.debug("_initialize_device: depth_frame_size " + str(depth_frame_size))
        logger.debug("_initialize_device: color_fps " + str(color_fps))
        logger.debug("_initialize_device: depth_fps " + str(depth_fps))
        color_frame_size = tuple(color_frame_size)
        depth_frame_size = tuple(depth_frame_size)

        self.last_color_frame_ts = None
        self.last_depth_frame_ts = None

        self._recent_frame = None
        self._recent_depth_frame = None

        if not devices:
            if not self._needs_restart:
                logger.error("Camera failed to initialize. No cameras connected.")
            self.device_id = None
            self.update_menu()
            return

        if self.pipeline_profile is not None and self.pipeline is not None:
            try:
                self.pipeline.stop()  # only call Device.stop() if its context
            except:
                logger.error(
                    "Device id is set ({}), but pipeline is not running.".format(
                        self.device_id
                    )
                )

        # use default streams to filter modes by rs_stream and rs_format
        self._available_modes = self._enumerate_formats(device_id)
        logger.debug(
            "_initialize_device: self._available_modes " + str(self._available_modes)
        )

        # verify that requested framesize-fps values are compatible
        if color_fps not in self._available_modes[rs.stream.color][color_frame_size]:
            old_fps = color_fps
            rates = [
                abs(r - color_fps)
                for r in self._available_modes[rs.stream.color][color_frame_size]
            ]
            best_rate_idx = rates.index(min(rates))
            color_fps = self._available_modes[rs.stream.color][color_frame_size][
                best_rate_idx
            ]
            logger.warning(
                "{} fps is not supported for ({}) for Color Stream. Fallback to {} fps".format(
                    old_fps, color_frame_size, color_fps
                )
            )

        if depth_fps not in self._available_modes[rs.stream.depth][depth_frame_size]:
            old_fps = depth_fps
            rates = [
                abs(r - depth_fps)
                for r in self._available_modes[rs.stream.depth][depth_frame_size]
            ]
            best_rate_idx = rates.index(min(rates))
            depth_fps = self._available_modes[rs.stream.depth][depth_frame_size][
                best_rate_idx
            ]
            logger.warning(
                "{} fps is not supported for ({}) for Depth Stream. Fallback to {} fps".format(
                    old_fps, depth_frame_size, depth_fps
                )
            )

        config = self._prep_configuration(
            color_frame_size, color_fps, depth_frame_size, depth_fps
        )

        try:
            self.pipeline_profile = self.pipeline.start(config)
        except RuntimeError as re:
            logger.error("Cannot start pipeline! " + str(re))
            self.pipeline_profile = None
        else:
            self.device_id = device_id
            self.streams = self.pipeline_profile.get_streams()
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

    # negative values mean "don't enable the stream"
    def _prep_configuration(
        self,
        color_frame_size=(-1, -1),
        color_fps=-1,
        depth_frame_size=(-1, -1),
        depth_fps=-1,
    ):
        config = rs.config()

        # only use these two formats
        color_format = rs.format.yuyv
        depth_format = rs.format.z16

        # FIXME this looks ugly
        if depth_frame_size[0] > 0 and depth_frame_size[1] > 0 and depth_fps > 0:
            config.enable_stream(
                rs.stream.depth,
                depth_frame_size[0],
                depth_frame_size[1],
                depth_format,
                depth_fps,
            )

        if color_frame_size[0] > 0 and color_frame_size[1] > 0 and color_fps > 0:
            config.enable_stream(
                rs.stream.color,
                color_frame_size[0],
                color_frame_size[1],
                color_format,
                color_fps,
            )

        return config

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
            serial = d.get_info(rs.camera_info.serial_number)
            if device_id == serial:
                current_device = d

        logger.debug("Found the current device: " + device_id)
        if current_device is None:
            return formats

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

                if stream_type not in formats:
                    formats[stream_type] = {}

                stream_resolution = (vp.width(), vp.height())

                if stream_resolution not in formats[stream_type]:
                    formats[stream_type][stream_resolution] = []
                formats[stream_type][stream_resolution].append(vp.fps())

        return formats

    def cleanup(self):
        if self.depth_video_writer is not None:
            self.stop_depth_recording()
        if self.pipeline is not None and self.pipeline_profile is not None:
            self.pipeline.stop()
            self.pipeline_profile = None

    def get_init_dict(self):
        return {"device_id": self.device_id}

    # raise NotImplementedError("get_init_dict requested")

    def get_frames(self):
        if self.pipeline and self.pipeline_profile:
            try:
                frames = self.pipeline.wait_for_frames(TIMEOUT)
            except RuntimeError as e:
                logger.error(
                    "Cannot wait for frames. Is the pipeline running? " + str(e)
                )
                return None, None
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
        if self._needs_restart:
            logger.debug("recent_events -> needs restart")
            self.restart_device()
            time.sleep(0.05)
        elif not self.online:
            logger.debug("recent_events -> not online!")
            time.sleep(0.05)
            return

        try:
            color_frame, depth_frame = self.get_frames()
        except RuntimeError as re:
            logger.warning(
                "Realsense failed to provide frames. Attempting to reinit. " + str(re)
            )
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

        if self.pipeline is None or self.pipeline_profile is None:
            self.menu.append(ui.Info_Text("Capture initialization failed."))
            return

        self.menu.append(ui.Switch("record_depth", self, label="Record Depth Stream"))
        self.menu.append(ui.Switch("preview_depth", self, label="Preview Depth"))

        color_sizes = sorted(self._available_modes[rs.stream.color], reverse=True)
        selector = ui.Selector(
            "frame_size",
            self,
            # setter=,
            selection=color_sizes,
            label="Color Resolution",
        )
        self.menu.append(selector)

        def color_fps_getter():

            avail_fps = [
                fps for fps in self._available_modes[rs.stream.color][self.frame_size]
            ]
            return avail_fps, [str(fps) for fps in avail_fps]

        selector = ui.Selector(
            "frame_rate",
            self,
            # setter=,
            selection_getter=color_fps_getter,
            label="Color Frame Rate",
        )
        self.menu.append(selector)

        depth_sizes = sorted(self._available_modes[rs.stream.depth], reverse=True)
        selector = ui.Selector(
            "depth_frame_size",
            self,
            # setter=,
            selection=depth_sizes,
            label="Depth Resolution",
        )
        self.menu.append(selector)

        def depth_fps_getter():
            avail_fps = [
                fps
                for fps in self._available_modes[rs.stream.depth][self.depth_frame_size]
            ]
            return avail_fps, [str(fps) for fps in avail_fps]

        selector = ui.Selector(
            "depth_frame_rate",
            self,
            selection_getter=depth_fps_getter,
            label="Depth Frame Rate",
        )
        self.menu.append(selector)

        def reset_options():
            raise NotImplementedError("update_menu::reset_options() requested")

        sensor_control = ui.Growing_Menu(label="Sensor Settings")
        sensor_control.append(
            ui.Button("Reset device options to default", reset_options)
        )
        for ctrl in sorted(self.controls.values(), key=lambda x: x.range.option):
            # sensor_control.append(ui.Info_Text(ctrl.description))
            if (
                ctrl.range.min == 0.0
                and ctrl.range.max == 1.0
                and ctrl.range.step == 1.0
            ):
                sensor_control.append(
                    ui.Switch("value", ctrl, label=ctrl.label, off_val=0.0, on_val=1.0)
                )
            else:
                sensor_control.append(
                    ui.Slider(
                        "value",
                        ctrl,
                        label=ctrl.label,
                        min=ctrl.range.min,
                        max=ctrl.range.max,
                        step=ctrl.range.step,
                    )
                )
        self.menu.append(sensor_control)

    def gl_display(self):
        if self.depth_window is not None and glfw.glfwWindowShouldClose(
            self.depth_window
        ):
            glfw.glfwDestroyWindow(self.depth_window)
            self.depth_window = None

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

    def restart_device(
        self,
        device_id=None,
        color_frame_size=None,
        color_fps=None,
        depth_frame_size=None,
        depth_fps=None,
        device_options=None,
    ):
        logger.debug("restart_device")
        if device_id is None:
            if self.pipeline_profile is not None:  # already running
                device_id = self.pipeline_profile.get_device().get_info(
                    rs.camera_info.serial_number
                )
            else:
                # set the first available device
                devices = self.context.query_devices()
                if devices:
                    device_id = devices[0].get_info(rs.camera_info.serial_number)
                else:
                    self.device_id = None
                    logger.error("Cannot restart device. No device connected.")
                    return

            self.device_id = device_id

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
        if self.pipeline is not None and self.pipeline_profile is not None:
            try:
                self.pipeline.stop()
                self.pipeline_profile = None
            except RuntimeError:
                logger.warning("Tried to stop self.pipeline before starting.")

        self.notify_all(
            {
                "subject": "realsense2_source.restart",
                "device_id": device_id,
                "color_frame_size": color_frame_size,
                "color_fps": color_fps,
                "depth_frame_size": depth_frame_size,
                "depth_fps": depth_fps,
                "device_options": device_options,
            }
        )

    def on_click(self, pos, button, action):
        pass

    def on_notify(self, notification):
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
    def frame_size(self):
        # logger.debug("get frame_size")
        try:
            stream_profile = self.stream_profiles[rs.stream.color]
            return stream_profile.width(), stream_profile.height()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created (color): {}".format(a))
        except KeyError as k:
            logger.error("Color stream is not found: {}".format(k))

    @frame_size.setter
    def frame_size(self, new_size):
        if new_size != self.frame_size:
            self.restart_device(color_frame_size=new_size)

    @property
    def frame_rate(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.color]
            return stream_profile.fps()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created: {}".format(a))
            return -1
        except KeyError as k:
            logger.error("Color stream is not found: {}".format(k))

    @frame_rate.setter
    def frame_rate(self, new_rate):
        if new_rate != self.frame_rate:
            self.restart_device(color_fps=new_rate)

    @property
    def depth_frame_size(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.depth]
            return stream_profile.width(), stream_profile.height()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created (depth): {}".format(a))
            return (-1, -1)
        except KeyError as k:
            logger.error("Depth stream is not found: {}".format(k))

    @depth_frame_size.setter
    def depth_frame_size(self, new_size):
        if new_size != self.depth_frame_size:
            self.restart_device(depth_frame_size=new_size)

    @property
    def depth_frame_rate(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.depth]
            return stream_profile.fps()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created: {}".format(a))
            return -1
        except KeyError as k:
            logger.error("Depth stream is not found: {}".format(k))

    @depth_frame_rate.setter
    def depth_frame_rate(self, new_rate):
        if new_rate != self.depth_frame_rate:
            self.restart_device(depth_fps=new_rate)

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        try:
            self.pipeline.start()
            self.pipeline.stop()
            return False
        except RuntimeError:
            return True

    @property
    def name(self):
        if self.pipeline_profile is not None:
            return self.pipeline_profile.get_device().get_info(rs.camera_info.name)
        else:
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
