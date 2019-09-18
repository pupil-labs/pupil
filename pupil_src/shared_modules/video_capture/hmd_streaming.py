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

import numpy as np
from pyglui import ui

import zmq_tools
from camera_models import Radial_Dist_Camera, Dummy_Camera
from video_capture.base_backend import Base_Manager, Base_Source

logger = logging.getLogger(__name__)


class RGBFrame:
    def __init__(self, buffer, timestamp, index, width, height):

        rgb = np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)
        self.bgr = np.ascontiguousarray(np.flip(rgb, (0, 2)))
        self.img = self.bgr
        self.timestamp = timestamp
        self.index = index
        self.width = width
        self.height = height
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None


FRAME_CLASS_BY_FORMAT = {"rgb": RGBFrame}


class HMD_Streaming_Source(Base_Source):
    name = "HMD Streaming"

    def __init__(self, g_pool, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        self.fps = 30
        self.projection_matrix = None
        self.frame_sub = zmq_tools.Msg_Receiver(
            self.g_pool.zmq_ctx,
            self.g_pool.ipc_sub_url,
            topics=("hmd_streaming.world",),
        )

    # def get_init_dict(self):

    def init_ui(self):  # was gui
        self.add_menu()
        self.menu.label = "HMD Streaming"
        text = ui.Info_Text("HMD Streaming Info")
        self.menu.append(text)

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.frame_sub = None

    def recent_events(self, events):
        frame = self.get_frame()
        if frame:
            events["frame"] = frame
            self._recent_frame = frame

    def get_frame(self):
        if self.frame_sub.socket.poll(timeout=50):  # timeout in ms (50ms -> 20fps)
            while self.frame_sub.new_data:  # drop all but the newest frame
                frame = self.frame_sub.recv()[1]

            try:
                frame_format = frame["format"]
                if frame_format in FRAME_CLASS_BY_FORMAT:
                    frame_class = FRAME_CLASS_BY_FORMAT[frame_format]
                    return self._process_frame(frame_class, frame)
            except KeyError as err:
                logger.debug(
                    "Ill-formatted frame received. Missing key: {}".format(err)
                )

    def _process_frame(self, frame_class, frame_data):
        projection_matrix = np.array(frame_data["projection_matrix"]).reshape(3, 3)
        if (projection_matrix != self.projection_matrix).any():
            self.projection_matrix = projection_matrix
            self._intrinsics = None  # resets intrinsics

        return frame_class(
            frame_data["__raw_data__"][0],
            frame_data["timestamp"],
            frame_data["index"],
            frame_data["width"],
            frame_data["height"],
        )

    @property
    def frame_size(self):
        return (
            (self._recent_frame.width, self._recent_frame.height)
            if self._recent_frame
            else (1280, 720)
        )

    @property
    def frame_rate(self):
        return self.fps

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return self._recent_frame is not None

    @property
    def intrinsics(self):
        if self._intrinsics is None or self._intrinsics.resolution != self.frame_size:
            if self.projection_matrix is not None:
                distortion = [[0.0, 0.0, 0.0, 0.0, 0.0]]
                self._intrinsics = Radial_Dist_Camera(
                    self.projection_matrix, distortion, self.frame_size, self.name
                )
            else:
                self._intrinsics = Dummy_Camera(self.frame_size, self.name)
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        logger.error(
            "HMD Streaming backend does not support setting intrinsics manually"
        )


class HMD_Streaming_Manager(Base_Manager):
    """Simple manager to explicitly activate a fake source"""

    gui_name = "HMD Streaming"

    def __init__(self, g_pool):
        super().__init__(g_pool)

    # Initiates the UI for starting the webcam.
    def init_ui(self):
        self.add_menu()
        from pyglui import ui

        self.menu.append(ui.Info_Text("Backend for HMD Streaming"))
        self.menu.append(ui.Button("Activate HMD Streaming", self.activate_source))

    def activate_source(self):
        settings = {}
        # if the user set fake capture, we dont want it to auto jump back to the old capture.
        if self.g_pool.process == "world":
            self.notify_all(
                {
                    "subject": "start_plugin",
                    "name": "HMD_Streaming_Source",
                    "args": settings,
                }
            )
        else:
            logger.warning("HMD Streaming backend is not supported in the eye process.")

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        pass

    def get_init_dict(self):
        return {}
