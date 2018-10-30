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


class Realsense2_Source(Base_Source):
    def __init__(
        self,
        g_pool,
        device_id=None,
        frame_size=(640, 480),
        frame_rate=30,
        depth_frame_size=(640, 480),
        depth_frame_rate=30,
        align_streams=False,
        preview_depth=False,
        device_options=(),
        record_depth=True,
        stream_preset=None,
    ):
        logger.info("_init_ started")
        super().__init__(g_pool)
        self._intrinsics = None
        self.color_frame_index = 0
        self.depth_frame_index = 0
        self.device_id = device_id  # we'll use serial_number for this
        self.context = rs.context()
        self.pipeline = rs.pipeline(self.context)
        self.pipeline_profile = None
        self.align_streams = align_streams
        self.preview_depth = preview_depth
        self.record_depth = record_depth
        self.depth_video_writer = None
        self.frame_size = frame_size
        self.depth_frame_size = depth_frame_size
        self.controls = {}
        self.pitch = 0
        self.yaw = 0
        self.last_pos = (0, 0)
        self.depth_window = None
        self._needs_restart = False
        self.stream_preset = stream_preset

        self._initialize_device(
            device_id,
            frame_size,
            frame_rate,
            depth_frame_size,
            depth_frame_rate,
            device_options,
        )
        logger.info("_init_ completed")

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
        print(color_frame_size)
        print(depth_frame_size)
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

        if self.device_id is not None:
            try:
                self.pipeline.stop()  # only call Device.stop() if its context
            except:
                logger.error(
                    "Device id is set ({}), but pipeline is not running.".format(
                        self.device_id
                    )
                )

        # use default streams to filter modes by rs_stream and rs_format
        self._available_modes = self._enumerate_formats(self.device_id)

        # FIXME we use some default settings at the time being
        color_frame_size = (640, 480)
        color_fps = 30
        depth_frame_size = (640, 480)
        depth_fps = 30
        # FIXME shoule be YUV? See class ColorFrame in realsense_backend.py
        color_format = rs.format.bgr8
        depth_format = rs.format.z16

        config = rs.config()
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

        self.pipeline_profile = self.pipeline.start(config)
        self.streams = self.pipeline_profile.get_streams()
        self.stream_profiles = {
            s.stream_type(): s for s in self.pipeline_profile.get_streams()
        }
        self.update_menu()
        self._needs_restart = False

    def _enumerate_formats(self, device_id):
        """Enumerate formats into hierachical structure:

        streams:
            resolutions:
                framerates
        """
        formats = {}

        # FIXME
        formats[rs.stream.color] = {(640, 480): [30]}
        formats[rs.stream.depth] = {(640, 480): [30]}

        return formats

    def cleanup(self):
        raise NotImplementedError("cleanup requested")

    def get_init_dict(self):
        raise NotImplementedError("get_init_dict requested")

    def get_frames(self):
        raise NotImplementedError("get_frames requested")

    def recent_events(self, events):
        pass

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Local USB Video Source"
        self.update_menu()

    def update_menu(self):
        try:
            del self.menu[:]
        except AttributeError:
            return

        from pyglui import ui

        if self.pipeline is None:
            self.menu.append(ui.Info_Text("Capture initialization failed."))
            return

        def align_and_restart(val):
            self.align_streams = val
            self.restart_device()

        self.menu.append(ui.Switch("record_depth", self, label="Record Depth Stream"))
        self.menu.append(ui.Switch("preview_depth", self, label="Preview Depth"))
        self.menu.append(
            ui.Switch(
                "align_streams", self, label="Align Streams", setter=align_and_restart
            )
        )

        native_presets = [("None", None)]

        def set_stream_preset(val):
            raise NotImplementedError("set_stream_preset requested")

        self.menu.append(
            ui.Selector(
                "stream_preset",
                self,
                setter=set_stream_preset,
                labels=[preset[0] for preset in native_presets],
                selection=[preset[1] for preset in native_presets],
                label="Stream preset",
            )
        )
        color_sizes = sorted(self._available_modes[rs.stream.color], reverse=True)
        selector = ui.Selector(
            "frame_size",
            self,
            # setter=,
            selection=color_sizes,
            label="Resolution" if self.align_streams else "Color Resolution",
        )
        selector.read_only = self.stream_preset is not None
        self.menu.append(selector)

        def color_fps_getter():
            avail_fps = [
                fps
                for fps in self._available_modes[rs.stream.color][self.frame_size]
                if self.depth_frame_rate % fps == 0
            ]
            return avail_fps, [str(fps) for fps in avail_fps]

        selector = ui.Selector(
            "frame_rate",
            self,
            # setter=,
            selection_getter=color_fps_getter,
            label="Color Frame Rate",
        )
        selector.read_only = self.stream_preset is not None
        self.menu.append(selector)

        if not self.align_streams:
            depth_sizes = sorted(self._available_modes[rs.stream.depth], reverse=True)
            selector = ui.Selector(
                "depth_frame_size",
                self,
                # setter=,
                selection=depth_sizes,
                label="Depth Resolution",
            )
            selector.read_only = self.stream_preset is not None
            self.menu.append(selector)

        def depth_fps_getter():
            avail_fps = [
                fps
                for fps in self._available_modes[rs.stream.depth][self.depth_frame_size]
                if fps % self.frame_rate == 0
            ]
            return avail_fps, [str(fps) for fps in avail_fps]

        selector = ui.Selector(
            "depth_frame_rate",
            self,
            selection_getter=depth_fps_getter,
            label="Depth Frame Rate",
        )
        selector.read_only = self.stream_preset is not None
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
        pass

    def restart_device(
        self,
        device_id=None,
        color_frame_size=None,
        color_fps=None,
        depth_frame_size=None,
        depth_fps=None,
        device_options=None,
    ):
        print("restart_device")
        if device_id is None:
            if self.pipeline_profile is not None:
                dev = self.pipeline_profile.get_device()
                self.device_id = dev.get_info(rs.camera_info.serial_number)
                # FIXME why do we need to refresh?
                device_id = self.device_id
            else:
                device_id = 0
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
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                logger.warning("Tried to stop self.pipeline before starting.")

        self.pipeline = rs.pipeline(self.context)
        self.pipeline.start()
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
        try:
            stream_profile = self.stream_profiles[rs.stream.color]
            stream_profile = stream_profile.as_video_stream_profile()
            return stream_profile.width(), stream_profile.height()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created (color): {}".format(a))
            return (-1, -1)
        except KeyError as k:
            logger.error("Color stream is not found: {}".format(k))

    @frame_size.setter
    def frame_size(self, new_size):
        if self.pipeline is not None and new_size != self.frame_size:
            self.restart_device(color_frame_size=new_size)

    @property
    def frame_rate(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.color]
            stream_profile = stream_profile.as_video_stream_profile()
            return stream_profile.fps()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created: {}".format(a))
            return -1
        except KeyError as k:
            logger.error("Color stream is not found: {}".format(k))

    @frame_rate.setter
    def frame_rate(self, new_rate):
        if self.pipeline is not None and new_rate != self.frame_rate:
            self.restart_device(color_fps=new_rate)

    @property
    def depth_frame_size(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.depth]
            stream_profile = stream_profile.as_video_stream_profile()
            return stream_profile.width(), stream_profile.height()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created (depth): {}".format(a))
            return (-1, -1)
        except KeyError as k:
            logger.error("Depth stream is not found: {}".format(k))

    @depth_frame_size.setter
    def depth_frame_size(self, new_size):
        if self.pipeline is not None and new_size != self.depth_frame_size:
            self.restart_device(depth_frame_size=new_size)

    @property
    def depth_frame_rate(self):
        try:
            stream_profile = self.stream_profiles[rs.stream.depth]
            stream_profile = stream_profile.as_video_stream_profile()
            return stream_profile.fps()
        except AttributeError as a:
            logger.info("Stream profiles are not yet created: {}".format(a))
            return -1
        except KeyError as k:
            logger.error("Depth stream is not found: {}".format(k))

    @depth_frame_rate.setter
    def depth_frame_rate(self, new_rate):
        if self.pipeline is not None and new_rate != self.depth_frame_rate:
            self.restart_device(depth_fps=new_rate)

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        try:  # FIXME: this does not necessarily means "streaming"
            self.pipeline.start()
            self.pipeline.stop()
            return False
        except RuntimeError:
            return True

    @property
    def name(self):
        # not the same as `if self.device:`!
        if self.pipeline_profile is not None:
            dev = self.pipeline_profile.get_device()
            return dev.get_info(rs.camera_info.name)
        else:
            return "Ghost capture"


class Realsense2_Manager(Base_Manager):
    """Manages Intel RealSense 3D sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """

    gui_name = "RealSense D400"

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()
        from pyglui import ui

        self.menu.append(ui.Info_Text("Intel RealSense 3D sources"))

        # FIXME see: https://github.com/IntelRealSense/librealsense/issues/2240#issuecomment-433105086
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
