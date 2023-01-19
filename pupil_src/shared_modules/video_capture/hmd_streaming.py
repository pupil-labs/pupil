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
import logging
from typing import Iterable, List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import zmq_tools
from camera_models import Dummy_Camera, Radial_Dist_Camera
from pyglui import ui
from typing_extensions import Literal, NotRequired, TypedDict
from video_capture.base_backend import Base_Source

logger = logging.getLogger(__name__)


class SerializedFrame(TypedDict):
    __raw_data__: List[bytes]
    timestamp: float
    index: int
    width: int
    height: int
    format: Literal["rgb", "bgr", "gray"]
    projection_matrix: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    distortion_coeffs: NotRequired[Tuple[float, ...]]


class Uint8BufferFrame(abc.ABC):
    def __init__(
        self,
        buffer: bytes,
        timestamp: float,
        index: int,
        width: int,
        height: int,
    ):
        #
        self._buffer = self.interpret_buffer(buffer, width, height)
        self.timestamp = timestamp
        self.index = index
        self.width = width
        self.height = height
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    def interpret_buffer(
        self, buffer: bytes, width: int, height: int
    ) -> npt.NDArray[np.uint8]:
        return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, self.depth)

    @property
    @abc.abstractmethod
    def depth(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gray(self) -> npt.NDArray[np.uint8]:  # dtype uint8, shape (height, width)
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def bgr(self) -> npt.NDArray[np.uint8]:
        # dtype uint8, shape (height, width, 3), memory needs to be allocated contiguous
        raise NotImplementedError

    @property
    def img(self) -> npt.NDArray[np.uint8]:
        # equivalent for bgr; kept for legacy reasons
        return self.bgr


class BGRFrame(Uint8BufferFrame):
    @property
    def depth(self) -> int:
        return 3

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
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
    def bgr(self) -> npt.NDArray[np.uint8]:
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
    def bgr(self) -> npt.NDArray[np.uint8]:
        try:
            return self._bgr
        except AttributeError:
            self._bgr = np.ascontiguousarray(np.dstack([self._buffer] * 3))
            return self._bgr

    @property
    def gray(self):
        return self._buffer

    def interpret_buffer(
        self, buffer: bytes, width: int, height: int
    ) -> npt.NDArray[np.uint8]:
        array = super().interpret_buffer(buffer, width, height)
        # since this will be our gray buffer, we need to get rid of our third dimension
        array.shape = height, width
        return array


FRAME_CLASS_BY_FORMAT = {"rgb": RGBFrame, "bgr": BGRFrame, "gray": GrayFrame}


class HMD_Streaming_Source(Base_Source):
    name = "HMD Streaming"

    def __init__(
        self,
        g_pool,
        topics: Iterable[str] = ("hmd_streaming.world",),
        hwm: int = 100,
        flip_preview: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        super().__init__(g_pool, *args, **kwargs)
        self.fps = 30
        self.projection_matrix = None
        self.distortion_coeffs = np.zeros((1, 5))
        self.__topics = topics
        self.__hwm = hwm
        self.frame_sub = zmq_tools.Msg_Receiver(
            self.g_pool.zmq_ctx,
            self.g_pool.ipc_sub_url,
            topics=self.__topics,
            hwm=self.__hwm,
        )
        if self.g_pool.process.startswith("eye") and flip_preview is not None:
            self._original_flip_value: Optional[bool] = self.g_pool.flip
            self.g_pool.flip = flip_preview
            logger.debug(f"Overwriting `g_pool.flip` to {flip_preview}")
        else:
            self._original_flip_value = None

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        init_dict["topics"] = self.__topics
        init_dict["hwm"] = self.__hwm
        return init_dict

    def cleanup(self):
        self.frame_sub = None
        if self._original_flip_value is not None:
            self.g_pool.flip = self._original_flip_value
            logger.debug(f"Restoring `g_pool.flip` to {self._original_flip_value}")

    def recent_events(self, events):
        frame = self.get_frame()
        if frame:
            events["frame"] = frame
            self._recent_frame = frame

    def get_frame(self):
        if self.frame_sub.socket.poll(timeout=50):  # timeout in ms (50ms -> 20fps)
            num_frames_dropped = -1
            frame: SerializedFrame = None
            while self.frame_sub.new_data:  # drop all but the newest frame
                num_frames_dropped += 1
                frame = self.frame_sub.recv()[1]
            if num_frames_dropped and self.g_pool.process == "world":
                logger.debug(f"Number of dropped frames: {num_frames_dropped}")

            try:
                frame_format = frame["format"]
                if frame_format in FRAME_CLASS_BY_FORMAT:
                    frame_class = FRAME_CLASS_BY_FORMAT[frame_format]
                    return self._process_frame(frame_class, frame)
            except KeyError as err:
                logger.debug(f"Ill-formatted frame received. Missing key: {err}")

    def _process_frame(
        self, frame_class: Type[Uint8BufferFrame], frame_data: SerializedFrame
    ):
        projection_matrix = np.array(frame_data["projection_matrix"]).reshape(3, 3)
        distortion_coeffs = np.array(
            frame_data.get("distortion_coeffs", (0.0,) * 5)
        ).reshape(1, -1)

        if (projection_matrix != self.projection_matrix).any() or (
            distortion_coeffs != self.distortion_coeffs
        ).any():
            self.projection_matrix = projection_matrix
            self.distortion_coeffs = distortion_coeffs
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
                self._intrinsics = Radial_Dist_Camera(
                    self.name,
                    self.frame_size,
                    self.projection_matrix,
                    self.distortion_coeffs,
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
        ui_elements.append(ui.Info_Text(self.name))
        return ui_elements
