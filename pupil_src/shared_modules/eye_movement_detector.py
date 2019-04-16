"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


# stdlib
import os
import abc
import csv
import enum
import typing
import logging
import traceback
import operator
import functools
import itertools
import collections
import bisect
from typing import _Protocol as Protocol


# local
from tasklib import interface
from tasklib.background.task import BackgroundGeneratorFunction
from tasklib.background.patches import Patch, IPCLoggingPatch
from tasklib.manager import PluginTaskManager
import file_methods as fm
import player_methods as pm

import methods
from plugin import Analysis_Plugin_Base
from observable import Observable
import video_capture as vc

# third-party
import nslr_hmm
import numpy as np
import cv2
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle
from pyglui.pyfontstash import fontstash


logger = logging.getLogger(__name__)

EYE_MOVEMENT_EVENT_KEY = 'eye_movement_segments'

# TODO: This protocol definition can be moved into `camera_models.py`
class Camera_Model(Protocol):
    def undistort(self, img: np.ndarray) -> np.ndarray:
        ...

    def unprojectPoints(
        self, pts_2d: np.ndarray, use_distortion: bool = True, normalize: bool = False
    ) -> np.ndarray:
        ...

    def projectPoints(
        self,
        object_points,
        rvec: typing.Optional[np.ndarray] = None,
        tvec: typing.Optional[np.ndarray] = None,
        use_distortion: bool = True,
    ):
        ...

    # def solvePnP(self, uv3d, xy): ...
    def save(self, directory: str, custom_name: typing.Optional[str] = None):
        ...


class Immutable_Capture:
    def __init__(self, capture: typing.Type[vc.base_backend.Base_Source]):
        self.frame_size: typing.Tuple[int, int] = (
            int(capture.frame_size[0]),
            int(capture.frame_size[1]),
        )
        self.intrinsics: Camera_Model = capture.intrinsics
        try:
            self.timestamps: np.ndarray = capture.timestamps
        except AttributeError:
            self.timestamps: np.ndarray = np.ndarray([])


Color_RGBA = typing.Tuple[float, float, float, float]

Color_RGB = typing.Tuple[int, int, int]

def color_rgb_to_rgba(rgb: Color_RGB) -> Color_RGBA:
    r, g, b = rgb
    return r/255, g/255, b/255, 1.0

@enum.unique
class Segment_Method(enum.Enum):
    GAZE = "gaze"
    PUPIL = "pupil"


@enum.unique
class Segment_Class(enum.Enum):
    FIXATION = "fixation"
    SACCADE = "saccade"
    POST_SACCADIC_OSCILLATIONS = "pso"
    SMOOTH_PURSUIT = "smooth_pursuit"

    @classmethod
    def from_nslr_class(cls, nslr_class):
        return {
            nslr_hmm.FIXATION: Segment_Class.FIXATION,
            nslr_hmm.SACCADE: Segment_Class.SACCADE,
            nslr_hmm.PSO: Segment_Class.POST_SACCADIC_OSCILLATIONS,
            nslr_hmm.SMOOTH_PURSUIT: Segment_Class.SMOOTH_PURSUIT,
        }[nslr_class]

    @property
    def color_rgba(self) -> Color_RGBA:
        return color_rgb_to_rgba(self.color_rgb)

    @property
    def color_rgb(self) -> Color_RGB:
        # https://flatuicolors.com/palette/defo
        grey = (52, 73, 94)  # wet asphalt
        yellow = (241, 196, 15)  # sun flower
        green = (39, 174, 96)  # nephritis
        blue = (41, 128, 185)  # belize hole
        purple = (142, 68, 173)  # wisteria
        return {
            Segment_Class.FIXATION: yellow,
            Segment_Class.SACCADE: green,
            Segment_Class.POST_SACCADIC_OSCILLATIONS: blue,
            Segment_Class.SMOOTH_PURSUIT: purple,
        }.get(self, grey)


Gaze_Data = typing.Iterable[fm.Serialized_Dict]


Gaze_Time = typing.Iterable[float]


MsgPack_Serialized_Segment = typing.Type[bytes]




class _Classified_Segment_Abstract_Storage(metaclass=abc.ABCMeta):
    def __getitem__(self, key):
        pass
    def get(self, key, default):
        pass
    def to_dict(self) -> dict:
        pass
    def to_serialized_dict(self) -> fm.Serialized_Dict:
        pass
    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        pass


class _Classified_Segment_Dict_Storage(_Classified_Segment_Abstract_Storage):
    def __init__(self, python_dict: dict):
        self._python_dict = python_dict
    def __getitem__(self, key):
        return self._python_dict.__getitem__(key)
    def get(self, key, default):
        return self._python_dict.get(key, default)
    def to_dict(self) -> dict:
        return self._python_dict
    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return fm.Serialized_Dict(python_dict=self._python_dict)
    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        serialized_dict = fm.Serialized_Dict(python_dict=self._python_dict)
        return serialized_dict.serialized


class _Classified_Segment_Serialized_Dict_Storage(_Classified_Segment_Abstract_Storage):
    def __init__(self, serialized_dict: fm.Serialized_Dict):
        self._serialized_dict = serialized_dict
    def __getitem__(self, key):
        return self._serialized_dict.__getitem__(key)
    def get(self, key, default):
        return self._serialized_dict.get(key, default)
    def to_dict(self) -> dict:
        return dict(self._serialized_dict)
    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return self._serialized_dict
    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        serialized_dict = self._serialized_dict
        return serialized_dict.serialized


class _Classified_Segment_MsgPack_Storage(_Classified_Segment_Serialized_Dict_Storage):
    def __init__(self, msgpack_bytes: MsgPack_Serialized_Segment):
        serialized_dict = fm.Serialized_Dict(msgpack_bytes=msgpack_bytes)
        super().__init__(serialized_dict=serialized_dict)




class Classified_Segment:
    def from_attrs(
        id: int,
        topic: str,
        use_pupil: bool,
        segment_data: Gaze_Data,
        segment_time: Gaze_Time,
        segment_class: Segment_Class,
        start_frame_index: int,
        end_frame_index: int,
        start_frame_timestamp: float,
        end_frame_timestamp: float,
    ) -> "Classified_Segment":

        segment_dict = {
            "id": id,
            "topic": topic,
            "use_pupil": use_pupil,
            "segment_data": segment_data,
            "segment_time": segment_time,
            "segment_class": segment_class.value,
            "start_frame_index": start_frame_index,
            "end_frame_index": end_frame_index,
            "start_frame_timestamp": start_frame_timestamp,
            "end_frame_timestamp": end_frame_timestamp,
        }

        segment_dict["confidence"] = float(
            np.mean([gp["confidence"] for gp in segment_data])
        )

        norm_pos_2d_points = np.array([gp["norm_pos"] for gp in segment_data])
        segment_dict["norm_pos"] = np.mean(norm_pos_2d_points, axis=0).tolist()

        if use_pupil:
            gaze_3d_points = np.array(
                [gp["gaze_point_3d"] for gp in segment_data], dtype=np.float32
            )
            segment_dict["gaze_point_3d"] = np.mean(gaze_3d_points, axis=0).tolist()

        return Classified_Segment.from_dict(segment_dict)

    def __init__(self, storage: typing.Type[_Classified_Segment_Abstract_Storage]):
        self._storage = storage

    def validate(self):
        assert self.frame_count > 0
        assert len(self.segment_data) == len(self.segment_time) == self.frame_count
        assert self.start_frame_timestamp <= self.end_frame_timestamp
        assert self.start_frame_timestamp == self.segment_time[0]
        assert self.end_frame_timestamp == self.segment_time[-1]

    # Serialization

    def to_dict(self) -> dict:
        return self._storage.to_dict()

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return self._storage.to_serialized_dict()

    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        return self._storage.to_msgpack()

    # Deserialization

    def from_dict(segment_dict: dict) -> "Classified_Segment":
        storage = _Classified_Segment_Dict_Storage(segment_dict)
        return Classified_Segment(storage)

    def from_serialized_dict(serialized_dict: fm.Serialized_Dict) -> "Classified_Segment":
        storage = _Classified_Segment_Serialized_Dict_Storage(serialized_dict)
        return Classified_Segment(storage)

    def from_msgpack(segment_msgpack: MsgPack_Serialized_Segment) -> "Classified_Segment":
        storage = _Classified_Segment_MsgPack_Storage(segment_msgpack)
        return Classified_Segment(storage)

    # Stored properties

    @property
    def id(self) -> int:
        """..."""
        return self._storage["id"]

    @property
    def topic(self) -> str:
        """..."""
        return self._storage["topic"]

    @property
    def use_pupil(self) -> bool:
        """..."""
        return self._storage["use_pupil"]

    @property
    def segment_data(self) -> list:  # FIXME: Annotate with typed list
        """..."""
        return self._storage["segment_data"]

    @property
    def segment_time(self) -> list:  # FIXME: Annotate with typed list
        """..."""
        return self._storage["segment_time"]

    @property
    def segment_class(self) -> Segment_Class:
        """..."""
        return Segment_Class(self._storage["segment_class"])

    @property
    def start_frame_index(self) -> int:
        """Index of the first segment frame, in the frame buffer."""
        return self._storage["start_frame_index"]

    @property
    def end_frame_index(self) -> int:
        """Index **after** the last segment frame, in the frame buffer."""
        return self._storage["end_frame_index"]

    @property
    def start_frame_timestamp(self) -> float:
        """Timestamp of the first frame, in the frame buffer."""
        return self._storage["start_frame_timestamp"]

    @property
    def end_frame_timestamp(self) -> float:
        """Timestamp of the last frame, in the frame buffer."""
        return self._storage["end_frame_timestamp"]

    @property
    def norm_pos(self):
        """..."""
        return self._storage["norm_pos"]

    @property
    def gaze_point_3d(self):
        """..."""
        return self._storage.get("gaze_point_3d", None)

    @property
    def confidence(self):
        """..."""
        return self._storage["confidence"]

    # Computed properties

    @property
    def method(self) -> Segment_Method:
        """..."""
        return Segment_Method.PUPIL if self.use_pupil else Segment_Method.GAZE

    @property
    def timestamp(self):
        """..."""
        return self.start_frame_timestamp

    @property
    def duration(self) -> float:
        """Duration in ms."""
        return (self.end_frame_timestamp - self.start_frame_timestamp) * 1000

    @property
    def frame_count(self) -> int:
        """..."""
        return self.end_frame_index - self.start_frame_index

    @property
    def mid_frame_index(self):
        """Index of the middle segment frame, in the frame buffer.
        """
        return int((self.end_frame_index + self.start_frame_index) // 2)

    @property
    def mid_frame_timestamp(self) -> float:
        """Timestamp of the middle frame, in the frame buffer."""
        return (self.end_frame_timestamp + self.start_frame_timestamp) / 2

    @property
    def color_rgb(self) -> Color_RGB:
        return self.segment_class.color_rgb

    @property
    def color_rgba(self) -> Color_RGBA:
        return self.segment_class.color_rgba

    def mean_2d_point_within_world(
        self, world_frame: typing.Tuple[int, int]
    ) -> typing.Tuple[int, int]:
        x, y = self.norm_pos
        x, y = methods.denormalize((x, y), world_frame, flip_y=True)
        return int(x), int(y)

    def last_2d_point_within_world(
        self, world_frame: typing.Tuple[int, int]
    ) -> typing.Tuple[int, int]:
        x, y = self.segment_data[-1]["norm_pos"]
        x, y = methods.denormalize((x, y), world_frame, flip_y=True)
        return int(x), int(y)

    def draw_on_frame(self, frame):
        #TODO: Type annotate frame
        world_frame = (frame.width, frame.height)
        segment_point = self.mean_2d_point_within_world(world_frame)
        segment_color = self.color_rgba

        pm.transparent_circle(
            frame.img, segment_point, radius=25.0, color=segment_color, thickness=3
        )

        text = str(self.id)
        text_origin = (segment_point[0] + 30, segment_point[1])
        text_fg_color = self.color_rgb
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        font_thickness = 1

        cv2.putText(
            img=frame.img,
            text=text,
            org=text_origin,
            fontFace=font_face,
            fontScale=font_scale,
            color=text_fg_color,
            thickness=font_thickness,
        )


class Classified_Segment_Factory:
    __slots__ = "_segment_id"

    def __init__(self, start_id: int = None):
        if start_id is None:
            start_id = 0
        assert isinstance(start_id, int)
        self._segment_id = start_id

    def create_segment(
        self, gaze_data, gaze_time, use_pupil, nslr_segment, nslr_segment_class
    ) -> typing.Optional[Classified_Segment]:
        id = self._get_id_postfix_increment()

        i_start, i_end = nslr_segment.i
        segment_data = list(gaze_data[i_start:i_end])
        segment_time = list(gaze_time[i_start:i_end])

        if len(segment_data) == 0:
            return None

        segment_class = Segment_Class.from_nslr_class(nslr_segment_class)
        topic = "nslr_segmentation"

        start_frame_index, end_frame_index = nslr_segment.i  # [i_0, i_1)
        start_frame_timestamp, end_frame_timestamp = (
            segment_time[0],
            segment_time[-1],
        )  # [t_0, t_1]

        segment = Classified_Segment.from_attrs(
            id=id,
            topic=topic,
            use_pupil=use_pupil,
            segment_data=segment_data,
            segment_time=segment_time,
            segment_class=segment_class,
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            start_frame_timestamp=start_frame_timestamp,
            end_frame_timestamp=end_frame_timestamp,
        )

        try:
            segment.validate()
        except:
            return None

        return segment

    def _get_id_postfix_increment(self) -> int:
        id = self._segment_id
        self._segment_id += 1
        return id


def can_use_3d_gaze_mapping(gaze_data) -> bool:
    return "gaze_normal_3d" in gaze_data[0] or "gaze_normals_3d" in gaze_data[0]


def clean_3d_data(gaze_points_3d: np.ndarray) -> np.ndarray:
    # Sometimes it is possible that the predicted gaze is behind the camera which is physically impossible.
    gaze_points_3d[gaze_points_3d[:, 2] < 0] *= -1.0
    return gaze_points_3d


def np_denormalize(points_2d, frame_size):
    width, height = frame_size
    points_2d[:, 0] *= width
    points_2d[:, 1] = (1.0 - points_2d[:, 1]) * height
    return points_2d


def preprocess_eye_movement_data(capture, gaze_data, use_pupil: bool):

    if use_pupil:
        assert can_use_3d_gaze_mapping(gaze_data)
        gaze_points_3d = [gp["gaze_point_3d"] for gp in gaze_data]
        gaze_points_3d = np.array(gaze_points_3d, dtype=np.float32)
        gaze_points_3d = clean_3d_data(gaze_points_3d)
    else:
        gaze_points_2d = np.array([gp["norm_pos"] for gp in gaze_data])
        gaze_points_2d = np_denormalize(gaze_points_2d, capture.frame_size)
        gaze_points_3d = capture.intrinsics.unprojectPoints(gaze_points_2d)

    x, y, z = gaze_points_3d.T
    r, theta, psi = methods.cart_to_spherical([x, y, z])

    nslr_data = np.column_stack([theta, psi])
    return nslr_data


Eye_Movement_Generator_Yield = typing.Tuple[
    str, typing.Optional[MsgPack_Serialized_Segment]
]


Eye_Movement_Generator = typing.Generator[Eye_Movement_Generator_Yield, None, None]


@typing.no_type_check
def eye_movement_detection_generator(
    capture: Immutable_Capture, gaze_data: Gaze_Data, factory_start_id: int = None
) -> Eye_Movement_Generator:
    yield "Preparing gaze data...", ()
    gaze_data = [
        fm.Serialized_Dict(msgpack_bytes=serialized) for serialized in gaze_data
    ]

    if not gaze_data:
        logger.warning("No data available to find fixations")
        yield "Segmentation complete", ()
        return

    use_pupil = can_use_3d_gaze_mapping(gaze_data)

    segment_factory = Classified_Segment_Factory(start_id=factory_start_id)

    gaze_time = np.array([gp["timestamp"] for gp in gaze_data])

    yield "Processing {} gaze data...".format("3d" if use_pupil else "2d"), ()
    eye_positions = preprocess_eye_movement_data(
        capture, gaze_data, use_pupil=use_pupil
    )

    yield "Classifying {} gaze data...".format("3d" if use_pupil else "2d"), ()
    sample_class, segmentation, classification = nslr_hmm.classify_gaze(
        gaze_time, eye_positions
    )

    yield "Detecting segmentation events...", ()
    for i, nslr_segment in enumerate(segmentation.segments):

        nslr_segment_class = classification[i]

        segment = segment_factory.create_segment(
            gaze_data=gaze_data,
            gaze_time=gaze_time,
            use_pupil=use_pupil,
            nslr_segment=nslr_segment,
            nslr_segment_class=nslr_segment_class,
        )

        if not segment:
            continue

        serialized = segment.to_msgpack()

        yield "Detecting segmentation events...", serialized

    yield "Segmentation complete", ()


GFY = typing.TypeVar("GFY")  # Generator function yield type
GFS = typing.TypeVar("GFS")  # Generator function send type
GFR = typing.TypeVar("GFR")  # Generator function return type

On_Started_Observer = typing.Callable[[], None]
On_Yield_Observer = typing.Callable[[GFY], None]
On_Completed_Observer = typing.Callable[[GFR], None]
On_Ended = typing.Callable[[], None]
On_Exception = typing.Callable[[Exception], None]
On_Canceled_Or_Killed = typing.Callable[[], None]

Eye_Movement_Detection_Task_Generator = typing.Generator[GFY, GFS, GFR]


class Eye_Movement_Detection_Task(
    BackgroundGeneratorFunction, typing.Generic[GFY, GFS, GFR]
):
    def __init__(
        self,
        name: str = "Segmentation",
        generator_function: Eye_Movement_Detection_Task_Generator = eye_movement_detection_generator,
        pass_shared_memory: bool = False,
        args: GFS = None,
        patches: typing.Iterable[typing.Type[Patch]] = tuple(),
    ):
        # TODO: A typed generic subclass of `BackgroundGeneratorFunction` can be added to `tasklib.background.task.py`
        super().__init__(
            name=name,
            generator_function=generator_function,
            pass_shared_memory=pass_shared_memory,
            args=args,
            kwargs={},
            patches=patches,
        )

    def add_observers(
        self,
        on_started: typing.Optional[On_Started_Observer] = None,
        on_yield: typing.Optional[On_Yield_Observer] = None,
        on_completed: typing.Optional[On_Completed_Observer] = None,
        on_ended: typing.Optional[On_Ended] = None,
        on_exception: typing.Optional[On_Exception] = None,
        on_canceled_or_killed: typing.Optional[On_Canceled_Or_Killed] = None,
    ):
        # TODO: A type-erased version of this method can be aaded to `TaskInterface`
        if on_started:
            self.add_observer("on_started", on_started)
        if on_yield:
            self.add_observer("on_yield", on_yield)
        if on_completed:
            self.add_observer("on_completed", on_completed)
        if on_ended:
            self.add_observer("on_ended", on_ended)
        if on_exception:
            self.add_observer("on_exception", on_exception)
        if on_canceled_or_killed:
            self.add_observer("on_canceled_or_killed", on_canceled_or_killed)


class Notification_Subject:
    SHOULD_RECALCULATE = "segmentation_detector.should_recalculate"
    SEGMENTATION_CHANGED = "segmentation_changed"


class _Seek_Notification_Subject:
    SHOULD_SEEK = "seek_control.should_seek"


class _Eye_Movement_Detector_Base(Analysis_Plugin_Base):
    icon_chr = chr(0xEC03)
    icon_font = "pupil_icons"


class Offline_Eye_Movement_Detector(Observable, _Eye_Movement_Detector_Base):
    """Eye movement classification detector based on segmented linear regression.
    """

    MENU_LABEL_TEXT = "Eye Movement Detector"

    def __init__(self, g_pool, show_segmentation=True):
        super().__init__(g_pool)
        self.show_segmentation = show_segmentation
        self.current_segment_index = None
        self.current_segment_details = None
        self.eye_movement_detection_yields = deque()
        self.status = ""

        self.task_manager = PluginTaskManager(self)
        self.eye_movement_task = None

        self.notify_all(
            {"subject": Notification_Subject.SHOULD_RECALCULATE, "delay": 0.5}
        )

    def init_ui(self):
        self.add_menu()
        self.menu.label = type(self).MENU_LABEL_TEXT

        def jump_next_segment(_):
            if len(self.g_pool.eye_movement_segments) < 1:
                logger.warning("No eye movement segments availabe")
                return

            # Set current segment index to next one, or to 0 if not available
            self.current_segment_index = (
                self.current_segment_index if self.current_segment_index else 0
            )
            self.current_segment_index = (self.current_segment_index + 1) % len(
                self.g_pool.eye_movement_segments
            )

            next_segment_ts = self.g_pool.eye_movement_segments[
                self.current_segment_index
            ].start_frame_timestamp

            self.notify_all(
                {
                    "subject": _Seek_Notification_Subject.SHOULD_SEEK,
                    "timestamp": next_segment_ts,
                }
            )

        def jump_prev_segment(_):
            if len(self.g_pool.eye_movement_segments) < 1:
                logger.warning("No segmentation availabe")
                return

            # Set current segment index to previous one, or to 0 if not available
            self.current_segment_index = (
                self.current_segment_index if self.current_segment_index else 0
            )
            self.current_segment_index = (self.current_segment_index - 1) % len(
                self.g_pool.eye_movement_segments
            )

            prev_segment_ts = self.g_pool.eye_movement_segments[
                self.current_segment_index
            ].start_frame_timestamp

            self.notify_all(
                {
                    "subject": _Seek_Notification_Subject.SHOULD_SEEK,
                    "timestamp": prev_segment_ts,
                }
            )

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(ui.Info_Text(help_str))

        self.menu.append(
            ui.Info_Text("Press the export button or type 'e' to start the export.")
        )

        detection_status_input = ui.Text_Input(
            "status", self, label="Detection progress:", setter=lambda x: None
        )

        show_segmentation_switch = ui.Switch(
            "show_segmentation", self, label="Show segmentation"
        )

        self.current_segment_details = ui.Info_Text("")

        self.next_segment_button = ui.Thumb(
            "jump_next_segment",
            setter=jump_next_segment,
            getter=lambda: False,
            label=chr(0xE044),
            hotkey="f",
            label_font="pupil_icons",
        )
        self.next_segment_button.status_text = "Next Segment"

        self.prev_segment_button = ui.Thumb(
            "jump_prev_segment",
            setter=jump_prev_segment,
            getter=lambda: False,
            label=chr(0xE045),
            hotkey="F",
            label_font="pupil_icons",
        )
        self.prev_segment_button.status_text = "Previous Segment"

        self.menu.append(detection_status_input)
        self.menu.append(show_segmentation_switch)
        self.menu.append(self.current_segment_details)

        self.g_pool.quickbar.append(self.next_segment_button)
        self.g_pool.quickbar.append(self.prev_segment_button)

    def deinit_ui(self):
        self.remove_menu()
        self.g_pool.quickbar.remove(self.next_segment_button)
        self.g_pool.quickbar.remove(self.prev_segment_button)
        self.current_segment_details = None
        self.next_segment_button = None
        self.prev_segment_button = None

    def get_init_dict(self):
        return {"show_segmentation": self.show_segmentation}

    def on_notify(self, notification):
        if notification["subject"] == "gaze_positions_changed":
            logger.info("Gaze postions changed. Recalculating.")
            self._classify()
        elif notification["subject"] == Notification_Subject.SHOULD_RECALCULATE:
            self._classify()
        elif notification["subject"] == "should_export":
            self.export_eye_movement(notification["range"], notification["export_dir"])

    def _classify(self):
        """
        classify eye movement
        """

        if self.g_pool.app == "exporter":
            return

        if self.eye_movement_task and self.eye_movement_task.running:
            self.eye_movement_task.kill(grace_period=1)

        capture = Immutable_Capture(self.g_pool.capture)
        gaze_data: Gaze_Data = [gp.serialized for gp in self.g_pool.gaze_positions]

        self.eye_movement_task = Eye_Movement_Detection_Task(args=(capture, gaze_data))
        self.task_manager.add_task(self.eye_movement_task)

        self.eye_movement_task.add_observers(
            on_started=self.on_task_started,
            on_yield=self.on_task_yield,
            on_completed=self.on_task_completed,
            on_ended=self.on_task_ended,
            on_exception=self.on_task_exception,
            on_canceled_or_killed=self.on_task_canceled_or_killed,
        )
        self.eye_movement_task.start()

    def on_task_started(self):
        self.eye_movement_detection_yields = collections.deque()

    def on_task_yield(self, yield_value):

        status, serialized = yield_value
        self.status = status

        if serialized:
            segment = Classified_Segment.from_msgpack(serialized)
            self.eye_movement_detection_yields.append(segment)

            current_ts = segment.end_frame_timestamp
            total_start_ts = self.g_pool.timestamps[0]
            total_end_ts = self.g_pool.timestamps[-1]

            current_duration = current_ts - total_start_ts
            total_duration = total_end_ts - total_start_ts

            progress = min(0.0, max(current_duration / total_duration, 1.0))
            self.menu_icon.indicator_stop = progress

    def on_task_exception(self, exception):
        pass

    def on_task_completed(self, return_value):
        self.status = "{} segments detected".format(
            len(self.eye_movement_detection_yields)
        )
        self.correlate_and_publish()

    def on_task_canceled_or_killed(self):
        pass

    def on_task_ended(self):
        if self.menu_icon:
            self.menu_icon.indicator_stop = 0.0

    def _ui_draw_visible_segments(self, frame, visible_segments):
        if not self.show_segmentation:
            return
        for segment in visible_segments:
            segment.draw_on_frame(frame)

    def _ui_update_segment_detail_text(self, index, total_count, focused_segment):

        if (index is None) or (total_count < 1) or (not focused_segment):
            self.current_segment_details.text = ""
            return

        info = ""
        prev_segment = (
            self.g_pool.eye_movement_segments[index - 1] if index > 0 else None
        )
        next_segment = (
            self.g_pool.eye_movement_segments[self.current_segment_index + 1]
            if self.current_segment_index < len(self.g_pool.eye_movement_segments) - 1
            else None
        )

        info += "Current segment, {} of {}\n".format(index + 1, total_count)
        info += "    ID: {}\n".format(focused_segment.id)
        info += "    Classification: {}\n".format(focused_segment.segment_class.value)
        info += "    Confidence: {:.2f}\n".format(focused_segment.confidence)
        info += "    Duration: {:.2f} milliseconds\n".format(focused_segment.duration)
        info += "    Frame range: {}-{}\n".format(
            focused_segment.start_frame_index + 1, focused_segment.end_frame_index + 1
        )
        info += "    2d gaze pos: x={:.3f}, y={:.3f}\n".format(
            *focused_segment.norm_pos
        )
        if focused_segment.gaze_point_3d:
            info += "    3d gaze pos: x={:.3f}, y={:.3f}, z={:.3f}\n".format(
                *focused_segment.gaze_point_3d
            )
        else:
            info += "    3d gaze pos: N/A\n"

        if prev_segment:
            info += "    Time since prev. segment: {:.2f} seconds\n".format(
                prev_segment.duration / 1000
            )
        else:
            info += "    Time since prev. segment: N/A\n"

        if next_segment:
            info += "    Time to next segment: {:.2f} seconds\n".format(
                focused_segment.duration / 1000
            )
        else:
            info += "    Time to next segment: N/A\n"
        self.current_segment_details.text = info

    def recent_events(self, events):

        frame = events.get("frame")
        if not frame:
            return

        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
        visible_segments: typing.Iterable[
            Classified_Segment
        ] = self.g_pool.eye_movement_segments.by_ts_window(frame_window)
        events[EYE_MOVEMENT_EVENT_KEY] = visible_segments

        def _find_focused_segment(visible_segments):
            current_segment = None
            visible_segments = visible_segments if visible_segments else []
            current_segment_index = self.current_segment_index

            if current_segment_index:
                current_segment_index = current_segment_index % len(
                    self.g_pool.eye_movement_segments
                )
                current_segment = self.g_pool.eye_movement_segments[
                    current_segment_index
                ]

            if not visible_segments:
                return current_segment_index, current_segment

            if (not current_segment in visible_segments) and len(visible_segments) > 0:
                current_segment = visible_segments[0]
                current_segment_index = self.g_pool.eye_movement_segments.data.index(
                    current_segment
                )

            return current_segment_index, current_segment

        self.current_segment_index, current_segment = _find_focused_segment(
            visible_segments
        )

        self._ui_draw_visible_segments(frame, visible_segments)
        self._ui_update_segment_detail_text(
            self.current_segment_index,
            len(self.g_pool.eye_movement_segments),
            current_segment,
        )

    def correlate_and_publish(self):
        self.g_pool.eye_movement_segments = pm.Affiliator(
            self.eye_movement_detection_yields,
            [
                segment.start_frame_timestamp
                for segment in self.eye_movement_detection_yields
            ],
            [
                segment.end_frame_timestamp
                for segment in self.eye_movement_detection_yields
            ],
        )
        self.notify_all(
            {"subject": Notification_Subject.SEGMENTATION_CHANGED, "delay": 1}
        )

    @classmethod
    def csv_schema(cls):
        return [
            ("id", lambda seg: seg.id),
            ("method", lambda seg: seg.method.value),
            ("segment_class", lambda seg: seg.segment_class.value),
            ("start_frame_index", lambda seg: seg.start_frame_index),
            ("end_frame_index", lambda seg: seg.end_frame_index),
            ("start_timestamp", lambda seg: seg.start_frame_timestamp),
            ("end_timestamp", lambda seg: seg.end_frame_timestamp),
            ("duration", lambda seg: seg.duration),
            ("confidence", lambda seg: seg.confidence),
            ("norm_pos_x", lambda seg: seg.norm_pos[0]),
            ("norm_pos_y", lambda seg: seg.norm_pos[1]),
            (
                "gaze_point_3d_x",
                lambda seg: seg.gaze_point_3d[0] if seg.gaze_point_3d else "",
            ),
            (
                "gaze_point_3d_y",
                lambda seg: seg.gaze_point_3d[1] if seg.gaze_point_3d else "",
            ),
            (
                "gaze_point_3d_z",
                lambda seg: seg.gaze_point_3d[2] if seg.gaze_point_3d else "",
            ),
        ]

    @classmethod
    def csv_header(cls):
        return tuple(label for label, _ in cls.csv_schema())

    @classmethod
    def csv_row(cls, segment):
        return tuple(str(getter(segment)) for _, getter in cls.csv_schema())

    def export_eye_movement(self, export_range, export_dir):

        if not self.eye_movement_detection_yields:
            logger.warning("No fixations in this recording nothing to export")
            return

        export_window = pm.exact_window(self.g_pool.timestamps, export_range)
        segments_in_section = self.g_pool.eye_movement_segments.by_ts_window(
            export_window
        )

        segment_export_filename = "eye_movement.csv"
        segment_export_full_path = os.path.join(export_dir, segment_export_filename)

        with open(
            segment_export_full_path, "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(type(self).csv_header())
            for segment in segments_in_section:
                csv_writer.writerow(type(self).csv_row(segment))
            logger.info("Created '{}' file.".format(segment_export_filename))
