"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import logging

import numpy as np
from pyglui import ui

import zmq_tools
from camera_models import Dummy_Camera, Radial_Dist_Camera
from video_capture.base_backend import Base_Source

logger = logging.getLogger(__name__)


class Uint8BufferFrame(abc.ABC):
    def __init__(self, buffer, timestamp, index, width, height):
        #
        self._buffer = self.interpret_buffer(buffer, width, height)
        self.timestamp = timestamp
        self.index = index
        self.width = width
        self.height = height
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    def interpret_buffer(self, buffer, width, height) -> np.ndarray:
        return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, self.depth)

    @property
    @abc.abstractmethod
    def depth(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gray(self) -> np.ndarray:  # dtype uint8, shape (height, width)
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def bgr(self) -> np.ndarray:
        # dtype uint8, shape (height, width, 3), memory needs to be allocated contiguous
        raise NotImplementedError

    @property
    def img(self) -> np.ndarray:  # equivalent for bgr; kept for legacy reasons
        return self.bgr


class BGRFrame(Uint8BufferFrame):
    @property
    def depth(self) -> int:
        return 3

    @property
    def bgr(self) -> np.ndarray:
        return self._buffer

    @property
    def gray(self):
        try:
            return self._gray
        except AttributeError:
            self._gray = np.mean(self._buffer, axis=-1).astype(self._buffer.dtype)
            return self._gray


class RGBFrame(BGRFrame):
    @property
    def bgr(self) -> np.ndarray:
        try:
            return self._bgr
        except AttributeError:
            self._bgr = np.ascontiguousarray(np.flip(self._buffer, (0, 2)))
            return self._bgr

    @property
    def gray(self):
        try:
            return self._gray
        except AttributeError:
            self._gray = np.mean(self._buffer, axis=-1).astype(self._buffer.dtype)
            return self._gray


class GrayFrame(Uint8BufferFrame):
    @property
    def depth(self) -> int:
        return 1

    @property
    def bgr(self) -> np.ndarray:
        try:
            return self._bgr
        except AttributeError:
            self._bgr = np.ascontiguousarray(np.dstack([self._buffer] * 3))
            return self._bgr

    @property
    def gray(self):
        return self._buffer

    def interpret_buffer(self, buffer, width, height) -> np.ndarray:
        buffer = super().interpret_buffer(buffer, width, height)
        # since this will be our gray buffer, we need to get rid of our third dimension
        buffer.shape = height, width
        return buffer


FRAME_CLASS_BY_FORMAT = {"rgb": RGBFrame, "bgr": BGRFrame, "gray": GrayFrame}


class HMD_Streaming_Source(Base_Source):
    name = "HMD Streaming"

    def __init__(self, g_pool, topics=("hmd_streaming.world",), hwm=1, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        self.fps = 30
        self.projection_matrix = None
        self.__topics = topics
        self.__hwm = hwm
        self.frame_sub = zmq_tools.Msg_Receiver(
            self.g_pool.zmq_ctx,
            self.g_pool.ipc_sub_url,
            topics=self.__topics,
            hwm=self.__hwm,
        )

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        init_dict["topics"] = self.__topics
        init_dict["hwm"] = self.__hwm
        return init_dict

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
                    self.name, self.frame_size, self.projection_matrix, distortion
                )
            else:
                self._intrinsics = Dummy_Camera(self.name, self.frame_size)
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        logger.error(
            "HMD Streaming backend does not support setting intrinsics manually"
        )

    def ui_elements(self):
        ui_elements = []
        ui_elements.append(ui.Info_Text(f"HMD Streaming"))
        return ui_elements
