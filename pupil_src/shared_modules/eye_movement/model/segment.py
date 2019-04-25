"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import enum
import typing as t
from gaze_producer.model.storage import StorageItem  # FIXME: Update import path when class moves to shared utils lib
from eye_movement.model.time_range import Time_Range
from eye_movement.model.color import Color, Defo_Color_Palette
import eye_movement.utils as utils
import methods as mt
import file_methods as fm
import player_methods as pm
import pyglui.cygl.utils as gl_utils
import numpy as np
import cv2
import nslr_hmm


@enum.unique
class Segment_Base_Type(enum.Enum):
    GAZE = "gaze"
    PUPIL = "pupil"


@enum.unique
class Segment_Class(enum.Enum):
    FIXATION = "fixation"
    SACCADE = "saccade"
    POST_SACCADIC_OSCILLATIONS = "pso"
    SMOOTH_PURSUIT = "smooth_pursuit"

    @staticmethod
    def from_nslr_class(nslr_class):
        return Segment_Class._nslr_class_to_segment_class_mapping[nslr_class]

    @property
    def color(self) -> Color:
        # TODO: Move drawing logic into ../ui/
        return self._segment_class_to_color_mapping[self]


# This needs to be defined outside class declaration; otherwise it is treated as a `enum` case.
Segment_Class._nslr_class_to_segment_class_mapping: t.Mapping[int, "Segment_Class"] = {
    nslr_hmm.FIXATION: Segment_Class.FIXATION,
    nslr_hmm.SACCADE: Segment_Class.SACCADE,
    nslr_hmm.PSO: Segment_Class.POST_SACCADIC_OSCILLATIONS,
    nslr_hmm.SMOOTH_PURSUIT: Segment_Class.SMOOTH_PURSUIT,
}


# TODO: Move drawing logic into ../ui/
# This needs to be defined outside class declaration; otherwise it is treated as a `enum` case.
Segment_Class._segment_class_to_color_mapping: t.Mapping["Segment_Class", t.Type[Color]] = {
    Segment_Class.FIXATION: Defo_Color_Palette.SUN_FLOWER,
    Segment_Class.SACCADE: Defo_Color_Palette.NEPHRITIS,
    Segment_Class.POST_SACCADIC_OSCILLATIONS: Defo_Color_Palette.BELIZE_HOLE,
    Segment_Class.SMOOTH_PURSUIT: Defo_Color_Palette.WISTERIA,
}


class Classified_Segment_Raw(abc.ABC):

    @abc.abstractmethod
    def __getitem__(self, key):
        ...

    @abc.abstractmethod
    def get(self, key, default):
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @abc.abstractmethod
    def to_dict(self) -> dict:
        ...

    @abc.abstractmethod
    def to_serialized_dict(self) -> fm.Serialized_Dict:
        ...

    @abc.abstractmethod
    def to_msgpack(self) -> utils.MsgPack_Serialized_Segment:
        ...

    def keys(self) -> t.Set[str]:
        return set(self)


class Classified_Segment_Raw_Dict(Classified_Segment_Raw):

    def __init__(self, python_dict: dict):
        self._python_dict = python_dict

    def __getitem__(self, key):
        return self._python_dict.__getitem__(key)

    def get(self, key, default):
        return self._python_dict.get(key, default)

    def __iter__(self):
        return self._python_dict.__iter__()

    def to_dict(self) -> dict:
        return self._python_dict

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return fm.Serialized_Dict(python_dict=self._python_dict)

    def to_msgpack(self) -> utils.MsgPack_Serialized_Segment:
        serialized_dict = fm.Serialized_Dict(python_dict=self._python_dict)
        return serialized_dict.serialized


class Classified_Segment_Raw_Serialized_Dict(Classified_Segment_Raw):

    def __init__(self, serialized_dict: fm.Serialized_Dict):
        self._serialized_dict = serialized_dict

    def __getitem__(self, key):
        return self._serialized_dict.__getitem__(key)

    def get(self, key, default):
        return self._serialized_dict.get(key, default)

    def __iter__(self):
        return self._serialized_dict.__iter__()

    def to_dict(self) -> dict:
        return dict(self._serialized_dict)

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return self._serialized_dict

    def to_msgpack(self) -> utils.MsgPack_Serialized_Segment:
        serialized_dict = self._serialized_dict
        return serialized_dict.serialized


class Classified_Segment_Raw_MsgPack(Classified_Segment_Raw_Serialized_Dict):

    def __init__(self, msgpack_bytes: utils.MsgPack_Serialized_Segment):
        serialized_dict = fm.Serialized_Dict(msgpack_bytes=msgpack_bytes)
        super().__init__(serialized_dict=serialized_dict)


class Classified_Segment(StorageItem):

    @staticmethod
    def from_attrs(
        id: int,
        topic: str,
        use_pupil: bool,
        segment_data: utils.Gaze_Data,
        segment_time: utils.Gaze_Time,
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

    def __init__(self, storage: Classified_Segment_Raw):
        self._storage = storage

    def validate(self):
        assert self.frame_count > 0
        assert len(self.segment_data) == len(self.segment_time) == self.frame_count
        assert self.start_frame_timestamp <= self.end_frame_timestamp
        assert self.start_frame_timestamp == self.segment_time[0]
        assert self.end_frame_timestamp == self.segment_time[-1]

    # Public Format

    def to_public_dict(self) -> dict:
        """
        Returns a dictionary representation of the segment,
        in the format suitable for sending over ZMQ,
        and consumption by clients external to this plugin.
        """

        public_dict = {
            "id": self.id,
            "topic": self.topic,
            "base_type": self.base_type.value,
            "segment_class": self.segment_class.value,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "start_frame_timestamp": self.start_frame_timestamp,
            "end_frame_timestamp": self.end_frame_timestamp,
        }
        return public_dict

    # Serialization

    def to_dict(self) -> dict:
        return self._storage.to_dict()

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return self._storage.to_serialized_dict()

    def to_msgpack(self) -> utils.MsgPack_Serialized_Segment:
        return self._storage.to_msgpack()

    # Deserialization

    @staticmethod
    def from_dict(segment_dict: dict) -> "Classified_Segment":
        storage = Classified_Segment_Raw_Dict(segment_dict)
        return Classified_Segment(storage)

    @staticmethod
    def from_serialized_dict(serialized_dict: fm.Serialized_Dict) -> "Classified_Segment":
        storage = Classified_Segment_Raw_Serialized_Dict(serialized_dict)
        return Classified_Segment(storage)

    @staticmethod
    def from_msgpack(segment_msgpack: utils.MsgPack_Serialized_Segment) -> "Classified_Segment":
        storage = Classified_Segment_Raw_MsgPack(segment_msgpack)
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
    def segment_data(self) -> t.List[fm.Serialized_Dict]:
        """..."""
        return self._storage["segment_data"]

    @property
    def segment_time(self) -> t.List[float]:
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
    def base_type(self) -> Segment_Base_Type:
        """..."""
        return Segment_Base_Type.PUPIL if self.use_pupil else Segment_Base_Type.GAZE

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
    def time_range(self) -> Time_Range:
        return Time_Range(start_time=self.start_frame_timestamp, end_time=self.end_frame_timestamp)

    def mean_2d_point_within_world(self, world_frame: t.Tuple[int, int]) -> t.Tuple[int, int]:
        x, y = self.norm_pos
        x, y = mt.denormalize((x, y), world_frame, flip_y=True)
        return int(x), int(y)

    def last_2d_point_within_world(self, world_frame: t.Tuple[int, int]) -> t.Tuple[int, int]:
        x, y = self.segment_data[-1]["norm_pos"]
        x, y = mt.denormalize((x, y), world_frame, flip_y=True)
        return int(x), int(y)

    @property
    def color(self) -> Color:
        # TODO: Move drawing logic into ../ui/
        return self.segment_class.color

    def draw_on_frame(self, frame):
        # TODO: Move drawing logic into ../ui/
        # TODO: Type annotate frame
        world_frame = (frame.width, frame.height)
        segment_point = self.mean_2d_point_within_world(world_frame)
        segment_color = self.color.to_rgba()

        pm.transparent_circle(
            frame.img, segment_point, radius=25.0, color=segment_color.channels, thickness=3
        )

        text = str(self.id)
        text_origin = (segment_point[0] + 30, segment_point[1])
        text_fg_color = self.color.to_rgb().channels
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

    def draw_in_gl_context(self, frame_size: t.Tuple[int, int], glfont):
        # TODO: Move drawing logic into ../ui/
        # TODO: Type annotate glfont
        segment_point = self.last_2d_point_within_world(frame_size)

        circle_color = self.color.to_rgba().channels
        gl_utils.draw_circle(
            segment_point,
            radius=48.0,
            stroke_width=10.0,
            color=gl_utils.RGBA(*circle_color)
        )

        font_size = 22
        text = str(self.id)
        text_origin_x = segment_point[0] + 48.0
        text_origin_y = segment_point[1]
        text_fg_color = self.color.to_rgba().channels

        glfont.set_size(font_size)
        glfont.set_color_float(text_fg_color)
        glfont.draw_text(text_origin_x, text_origin_y, text)


class Classified_Segment_Factory:
    __slots__ = "_segment_id"

    def __init__(self, start_id: int = None):
        if start_id is None:
            start_id = 0
        assert isinstance(start_id, int)
        self._segment_id = start_id

    def create_segment(
        self, gaze_data, gaze_time, use_pupil, nslr_segment, nslr_segment_class
    ) -> t.Optional[Classified_Segment]:
        segment_id = self._get_id_postfix_increment()

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
            id=segment_id,
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
        except AssertionError:
            return None

        return segment

    def _get_id_postfix_increment(self) -> int:
        segment_id = self._segment_id
        self._segment_id += 1
        return segment_id
