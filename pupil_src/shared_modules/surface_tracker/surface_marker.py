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
import typing
import functools
import collections

import cv2
import numpy as np

from pupil_src.shared_modules.apriltag import DetectionBase as Apriltag_V2_Detection


__all__ = [
    "Surface_Marker",
    "Surface_Marker_UID",
    "Surface_Marker_TagID",
    "Surface_Marker_Type",
]


Surface_Marker_UID = typing.NewType("Surface_Marker_UID", str)


Surface_Marker_TagID = typing.NewType("Surface_Marker_TagID", int)


@enum.unique
class Surface_Marker_Type(enum.Enum):
    # TODO: Is there a better (more uniquely descriptive) name than "square"?
    SQUARE = "square"
    APRILTAG_V2 = "apriltag_v2"


class Surface_Base_Marker(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def from_tuple(state: tuple) -> "Surface_Base_Marker":
        pass

    @abc.abstractmethod
    def to_tuple(self) -> tuple:
        pass

    @property
    @abc.abstractmethod
    def uid(self) -> Surface_Marker_UID:
        """
        Identifier that is guaranteed to be unique accross different marker types and tag families.
        """
        pass

    @property
    @abc.abstractmethod
    def tag_id(self) -> Surface_Marker_TagID:
        """
        Tag identifier that is unique only within the same marker type and tag family.
        """
        pass

    @property
    @abc.abstractmethod
    def id_confidence(self) -> float:
        # TODO: Why is it called "id_confidence" instead of "confidence"?
        pass

    @property
    @abc.abstractmethod
    def verts_px(self) -> list:
        pass

    @property
    @abc.abstractmethod
    def perimeter(self) -> float:
        # TODO: Is this used/useful outside surface_marker_detector.py? If not - remove
        pass

    @property
    @abc.abstractmethod
    def marker_type(self) -> Surface_Marker_Type:
        pass


# TODO: This object should offer a mean/centroid function to be used when drawing the marker toggle buttons
_Square_Marker_Detection_Raw = collections.namedtuple(
    "Square_Marker_Detection",
    ["raw_id", "id_confidence", "verts_px", "perimeter", "raw_marker_type"],
)


class _Square_Marker_Detection(_Square_Marker_Detection_Raw, Surface_Base_Marker):
    __slots__ = ()

    marker_type = Surface_Marker_Type.SQUARE

    @staticmethod
    def from_tuple(state: tuple) -> "_Square_Marker_Detection":
        cls = _Square_Marker_Detection
        expected_marker_type = cls.marker_type

        assert len(state) > 0
        assert isinstance(state[-1], str)

        try:
            real_marker_type = Surface_Marker_Type(state[-1])
        except ValueError:
            real_marker_type = expected_marker_type
            state = (*state, real_marker_type.value)

        assert real_marker_type == expected_marker_type
        return cls(*state)

    def to_tuple(self) -> tuple:
        return tuple(self)

    @property
    def uid(self) -> Surface_Marker_UID:
        marker_type = self.marker_type.value
        tag_id = self.tag_id
        return Surface_Marker_UID(f"{marker_type}:{tag_id}")

    @property
    def tag_id(self) -> Surface_Marker_TagID:
        return Surface_Marker_TagID(int(self.raw_id))


_Apriltag_V2_Marker_Detection_Raw = collections.namedtuple(
    "Apriltag_V2_Marker_Detection",
    [
        "tag_family",
        "raw_id",
        "hamming",
        "goodness",
        "decision_margin",
        "homography",
        "center",
        "corners",
        "raw_marker_type",
    ],
)


class _Apriltag_V2_Marker_Detection(
    _Apriltag_V2_Marker_Detection_Raw, Surface_Base_Marker
):
    __slots__ = ()

    marker_type = Surface_Marker_Type.APRILTAG_V2

    @staticmethod
    def from_tuple(state: tuple) -> "_Apriltag_V2_Marker_Detection":
        cls = _Apriltag_V2_Marker_Detection
        expected_marker_type = cls.marker_type

        assert len(state) > 0
        assert isinstance(state[-1], str)

        try:
            real_marker_type = Surface_Marker_Type(state[-1])
        except ValueError:
            real_marker_type = expected_marker_type
            state = (*state, real_marker_type.value)

        assert real_marker_type == expected_marker_type
        return cls(*state)

    def to_tuple(self) -> tuple:
        return tuple(self)

    @property
    def uid(self) -> Surface_Marker_UID:
        marker_type = self.marker_type.value
        tag_family = self.tag_family
        tag_id = self.tag_id
        return Surface_Marker_UID(f"{marker_type}:{tag_family}:{tag_id}")

    @property
    def tag_id(self) -> Surface_Marker_TagID:
        return Surface_Marker_TagID(int(self.raw_id))

    @property
    def id_confidence(self) -> float:
        decision_margin = self.decision_margin
        decision_margin /= 100.0
        decision_margin = max(0.0, min(decision_margin, 1.0))
        # TODO: Not sure this is the best estimate of confidence, and if decision_margin is in (0, 100)
        return decision_margin

    @property
    def verts_px(self) -> list:
        # Wrapping each point in a list is needed for compatibility with square detector
        return [[point] for point in self.corners]

    @property
    def perimeter(self) -> float:
        verts_px = np.asarray(self.verts_px, dtype=np.float32)
        perimeter = cv2.arcLength(verts_px, closed=True)
        return perimeter


# This exists because there is no easy way to make a user-defined class serializable with msgpack without extra hooks.
# Therefore, Surface_Marker is defined as a sublcass of _Raw_Surface_Marker, which is a subclass of namedtuple,
# because msgpack is able to serialize namedtuple subclasses out of the box.
_Surface_Marker_Raw = collections.namedtuple("Surface_Marker", ["raw_marker"])


class Surface_Marker(_Surface_Marker_Raw, Surface_Base_Marker):
    @staticmethod
    def from_square_tag_detection(detection: dict) -> "Surface_Marker":
        cls = _Square_Marker_Detection
        raw_marker = cls(
            raw_id=detection["id"],
            id_confidence=detection["id_confidence"],
            verts_px=detection["verts"],
            perimeter=detection["perimeter"],
            raw_marker_type=cls.marker_type.value,
        )
        return Surface_Marker(raw_marker=raw_marker)

    @staticmethod
    def from_apriltag_v2_detection(
        detection: Apriltag_V2_Detection
    ) -> "Surface_Marker":
        cls = _Apriltag_V2_Marker_Detection
        raw_marker = cls(
            tag_family=detection.tag_family.decode("utf8"),
            raw_id=detection.tag_id,
            hamming=detection.hamming,
            goodness=detection.goodness,
            decision_margin=detection.decision_margin,
            homography=detection.homography.tolist(),
            center=detection.center.tolist(),
            corners=detection.corners.tolist(),
            raw_marker_type=cls.marker_type.value,
        )
        return Surface_Marker(raw_marker=raw_marker)

    @staticmethod
    def deserialize(args) -> "Surface_Marker":
        if isinstance(args, list) and len(args) == 1:
            state = tuple(*args)
        else:
            state = tuple(args)
        return Surface_Marker.from_tuple(state=state)

    @staticmethod
    def from_tuple(state: tuple) -> "Surface_Marker":
        marker_type = state[-1]
        if marker_type == _Square_Marker_Detection.marker_type.value:
            raw_marker = _Square_Marker_Detection.from_tuple(state)
        elif marker_type == _Apriltag_V2_Marker_Detection.marker_type.value:
            raw_marker = _Apriltag_V2_Marker_Detection.from_tuple(state)
        else:
            raw_marker_type = _Square_Marker_Detection.marker_type.value
            raw_marker = _Square_Marker_Detection(*state, raw_marker_type)  # Legacy
        assert raw_marker is not None
        return Surface_Marker(raw_marker=raw_marker)

    def to_tuple(self) -> tuple:
        state = tuple(self)
        state = (*state, self.marker_type)
        return state

    @property
    def uid(self) -> Surface_Marker_UID:
        return self.raw_marker.uid

    @property
    def tag_id(self) -> Surface_Marker_TagID:
        return self.raw_marker.tag_id

    @property
    def id_confidence(self) -> float:
        return self.raw_marker.id_confidence

    @property
    def verts_px(self) -> list:
        return self.raw_marker.verts_px

    @property
    def perimeter(self) -> float:
        return self.raw_marker.perimeter

    @property
    def marker_type(self) -> Surface_Marker_Type:
        return self.raw_marker.marker_type


def test_surface_marker_from_raw_detection():
    # TODO: Test `from_*_detection` methods below
    # Surface_Marker.from_square_tag_detection({})
    # Surface_Marker.from_apriltag_v2_detection(apriltag.DetectionBase())
    assert True


def test_surface_marker_deserialize():

    # Square tag deserialization test

    SQUARE_MARKER_TAG_ID = 55
    SQUARE_MARKER_CONF = 0.0039215686274509665
    SQUARE_MARKER_VERTS = [
        [[1084.0, 186.0]],
        [[1089.0, 198.0]],
        [[1099.0, 195.0]],
        [[1095.0, 184.0]],
    ]
    SQUARE_MARKER_PERIM = 46.32534599304199

    # This is the format in which old (before Apriltag support was added) square tags where represented when serialized to msgpack
    old_serialized_square = [
        SQUARE_MARKER_TAG_ID,
        SQUARE_MARKER_CONF,
        SQUARE_MARKER_VERTS,
        SQUARE_MARKER_PERIM,
    ]
    # This is the format in which new square tags are represented when serialized to msgpack
    new_serialized_square = [
        [
            SQUARE_MARKER_TAG_ID,
            SQUARE_MARKER_CONF,
            SQUARE_MARKER_VERTS,
            SQUARE_MARKER_PERIM,
            _Square_Marker_Detection.marker_type.value,
        ]
    ]

    # Both formats should be supported by `Surface_Marker.deserialize` for backwards compatibility
    old_marker_square = Surface_Marker.deserialize(old_serialized_square)
    new_marker_square = Surface_Marker.deserialize(new_serialized_square)

    assert old_marker_square.marker_type == Surface_Marker_Type.SQUARE
    assert old_marker_square.tag_id == SQUARE_MARKER_TAG_ID
    assert old_marker_square.id_confidence == SQUARE_MARKER_CONF
    assert old_marker_square.verts_px == SQUARE_MARKER_VERTS
    assert old_marker_square.perimeter == SQUARE_MARKER_PERIM

    assert new_marker_square.marker_type == old_marker_square.marker_type
    assert new_marker_square.tag_id == old_marker_square.tag_id
    assert new_marker_square.id_confidence == old_marker_square.id_confidence
    assert new_marker_square.verts_px == old_marker_square.verts_px
    assert new_marker_square.perimeter == old_marker_square.perimeter

    # Apriltag V2 deserialization test

    APRILTAG_V2_FAMILY = "tag36h11"
    APRILTAG_V2_TAG_ID = 10
    APRILTAG_V2_HAMMING = 2
    APRILTAG_V2_GOODNESS = 0.0
    APRILTAG_V2_DECISION_MARGIN = 44.26249694824219
    APRILTAG_V2_HOMOGRAPHY = [
        [0.7398546228643903, 0.24224258644348548, -22.823628761428765],
        [-0.14956381555697143, -0.595697080889624, -53.73760032443805],
        [0.00036910994224440203, -0.001201257450114289, -0.07585102600797115],
    ]
    APRILTAG_V2_CENTER = [300.90072557529066, 708.4624052256166]
    APRILTAG_V2_CORNERS = [
        [317.3298034667968, 706.38671875],
        [300.56298828125, 717.4339599609377],
        [284.8282165527345, 710.4930419921874],
        [301.2247619628906, 699.854797363281],
    ]

    new_serialized_apriltag_v2 = [
        [
            APRILTAG_V2_FAMILY,
            APRILTAG_V2_TAG_ID,
            APRILTAG_V2_HAMMING,
            APRILTAG_V2_GOODNESS,
            APRILTAG_V2_DECISION_MARGIN,
            APRILTAG_V2_HOMOGRAPHY,
            APRILTAG_V2_CENTER,
            APRILTAG_V2_CORNERS,
            _Apriltag_V2_Marker_Detection.marker_type.value,
        ]
    ]

    new_marker_apriltag_v2 = Surface_Marker.deserialize(new_serialized_apriltag_v2)

    APRILTAG_V2_CONF = APRILTAG_V2_DECISION_MARGIN / 100
    APRILTAG_V2_VERTS = [[c] for c in APRILTAG_V2_CORNERS]
    APRILTAG_V2_PERIM = 74.20

    assert new_marker_apriltag_v2.marker_type == Surface_Marker_Type.APRILTAG_V2
    assert new_marker_apriltag_v2.tag_id == APRILTAG_V2_TAG_ID
    assert new_marker_apriltag_v2.id_confidence == APRILTAG_V2_CONF
    assert new_marker_apriltag_v2.verts_px == APRILTAG_V2_VERTS
    assert round(new_marker_apriltag_v2.perimeter, 2) == APRILTAG_V2_PERIM


if __name__ == "__main__":
    test_surface_marker_from_raw_detection()
    test_surface_marker_deserialize()
