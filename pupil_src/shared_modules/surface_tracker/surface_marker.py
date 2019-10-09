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
import itertools
import collections

import cv2
import numpy as np

from pupil_apriltags import Detection as Apriltag_V3_Detection


__all__ = [
    "Surface_Marker",
    "Surface_Marker_UID",
    "Surface_Marker_TagID",
    "Surface_Marker_Type",
    "create_surface_marker_uid",
    "parse_surface_marker_type",
    "parse_surface_marker_tag_id",
    "parse_surface_marker_tag_family",
]


Surface_Marker_UID = typing.NewType("Surface_Marker_UID", str)


Surface_Marker_TagID = typing.NewType("Surface_Marker_TagID", int)


@enum.unique
class Surface_Marker_Type(enum.Enum):
    SQUARE = "legacy"
    APRILTAG_V3 = "apriltag_v3"


def parse_surface_marker_type(uid: Surface_Marker_UID) -> Surface_Marker_Type:
    marker_type, _, _ = _parse_surface_marker_uid_components(uid=uid)
    return marker_type


def parse_surface_marker_tag_family(uid: Surface_Marker_UID) -> typing.Optional[str]:
    _, tag_family, _ = _parse_surface_marker_uid_components(uid=uid)
    return tag_family


def parse_surface_marker_tag_id(uid: Surface_Marker_UID) -> Surface_Marker_TagID:
    _, _, tag_id = _parse_surface_marker_uid_components(uid=uid)
    return tag_id


def create_surface_marker_uid(
    marker_type: Surface_Marker_Type,
    tag_family: typing.Optional[str],
    tag_id: Surface_Marker_TagID,
) -> Surface_Marker_UID:
    marker_type = marker_type.value
    if tag_family is None:
        return Surface_Marker_UID(f"{marker_type}:{tag_id}")
    else:
        return Surface_Marker_UID(f"{marker_type}:{tag_family}:{tag_id}")


def _parse_surface_marker_uid_components(
    uid: Surface_Marker_UID
) -> typing.Tuple[Surface_Marker_Type, typing.Optional[str], Surface_Marker_TagID]:
    components = str(uid).split(":")
    if len(components) == 2:
        marker_type, tag_id = components
        tag_family = None
    elif len(components) == 3:
        marker_type, tag_family, tag_id = components
    else:
        raise ValueError(f'Invalid surface marker uid: "{uid}"')
    return (
        Surface_Marker_Type(marker_type),
        tag_family,
        Surface_Marker_TagID(int(tag_id)),
    )


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
        """
        Confidence that the surface marker has the suggested `tag_id`.
        """
        pass

    @property
    @abc.abstractmethod
    def verts_px(self) -> list:
        pass

    @property
    @abc.abstractmethod
    def perimeter(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def marker_type(self) -> Surface_Marker_Type:
        pass

    def centroid(self) -> typing.Tuple[float, float]:
        centroid = np.mean(self.verts_px, axis=0)
        centroid = tuple(*centroid.tolist())
        return centroid


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
        return create_surface_marker_uid(
            marker_type=self.marker_type, tag_family=None, tag_id=self.tag_id
        )

    @property
    def tag_id(self) -> Surface_Marker_TagID:
        return Surface_Marker_TagID(int(self.raw_id))


_Apriltag_V3_Marker_Detection_Raw = collections.namedtuple(
    "Apriltag_V3_Marker_Detection",
    [
        "tag_family",
        "raw_id",
        "hamming",
        "decision_margin",
        "homography",
        "center",
        "corners",
        "pose_R",
        "pose_t",
        "pose_err",
        "raw_marker_type",
    ],
)


class _Apriltag_V3_Marker_Detection(
    _Apriltag_V3_Marker_Detection_Raw, Surface_Base_Marker
):
    __slots__ = ()

    marker_type = Surface_Marker_Type.APRILTAG_V3

    @staticmethod
    def from_tuple(state: tuple) -> "_Apriltag_V3_Marker_Detection":
        cls = _Apriltag_V3_Marker_Detection
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
        return create_surface_marker_uid(
            marker_type=self.marker_type, tag_family=self.tag_family, tag_id=self.tag_id
        )

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
    def from_apriltag_v3_detection(
        detection: Apriltag_V3_Detection
    ) -> "Surface_Marker":
        cls = _Apriltag_V3_Marker_Detection
        raw_marker = cls(
            tag_family=detection.tag_family.decode("utf8"),
            raw_id=detection.tag_id,
            hamming=detection.hamming,
            decision_margin=detection.decision_margin,
            homography=detection.homography.tolist(),
            center=detection.center.tolist(),
            corners=detection.corners.tolist(),
            pose_R=detection.pose_R,
            pose_t=detection.pose_t,
            pose_err=detection.pose_err,
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
        elif marker_type == _Apriltag_V3_Marker_Detection.marker_type.value:
            raw_marker = _Apriltag_V3_Marker_Detection.from_tuple(state)
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
    # Surface_Marker.from_apriltag_v3_detection(pupil_apriltags.Detection())
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

    # Apriltag V3 deserialization test

    APRILTAG_V3_FAMILY = "tag36h11"
    APRILTAG_V3_TAG_ID = 10
    APRILTAG_V3_HAMMING = 2
    APRILTAG_V3_DECISION_MARGIN = 44.26249694824219
    APRILTAG_V3_HOMOGRAPHY = [
        [0.7398546228643903, 0.24224258644348548, -22.823628761428765],
        [-0.14956381555697143, -0.595697080889624, -53.73760032443805],
        [0.00036910994224440203, -0.001201257450114289, -0.07585102600797115],
    ]
    APRILTAG_V3_CENTER = [300.90072557529066, 708.4624052256166]
    APRILTAG_V3_CORNERS = [
        [317.3298034667968, 706.38671875],
        [300.56298828125, 717.4339599609377],
        [284.8282165527345, 710.4930419921874],
        [301.2247619628906, 699.854797363281],
    ]
    APRILTAG_V3_POSE_R = None
    APRILTAG_V3_POSE_T = None
    APRILTAG_V3_POSE_ERR = None

    new_serialized_apriltag_v3 = [
        [
            APRILTAG_V3_FAMILY,
            APRILTAG_V3_TAG_ID,
            APRILTAG_V3_HAMMING,
            APRILTAG_V3_DECISION_MARGIN,
            APRILTAG_V3_HOMOGRAPHY,
            APRILTAG_V3_CENTER,
            APRILTAG_V3_CORNERS,
            APRILTAG_V3_POSE_R,
            APRILTAG_V3_POSE_T,
            APRILTAG_V3_POSE_ERR,
            _Apriltag_V3_Marker_Detection.marker_type.value,
        ]
    ]

    new_marker_apriltag_v3 = Surface_Marker.deserialize(new_serialized_apriltag_v3)

    APRILTAG_V3_CONF = APRILTAG_V3_DECISION_MARGIN / 100
    APRILTAG_V3_VERTS = [[c] for c in APRILTAG_V3_CORNERS]
    APRILTAG_V3_PERIM = 74.20

    assert new_marker_apriltag_v3.marker_type == Surface_Marker_Type.APRILTAG_V3
    assert new_marker_apriltag_v3.tag_id == APRILTAG_V3_TAG_ID
    assert new_marker_apriltag_v3.id_confidence == APRILTAG_V3_CONF
    assert new_marker_apriltag_v3.verts_px == APRILTAG_V3_VERTS
    assert round(new_marker_apriltag_v3.perimeter, 2) == APRILTAG_V3_PERIM


def test_surface_marker_uid_helpers():
    all_marker_types = set(Surface_Marker_Type)
    all_tag_ids = [Surface_Marker_TagID(123)]
    all_tag_families = ["best_tags", None]
    all_combinations = itertools.product(
        all_marker_types, all_tag_families, all_tag_ids
    )

    for marker_type, tag_family, tag_id in all_combinations:
        uid = create_surface_marker_uid(
            marker_type=marker_type, tag_family=tag_family, tag_id=tag_id
        )
        assert len(uid) > 0, "Surface_Marker_UID is not valid"

        assert parse_surface_marker_type(uid=uid) == marker_type
        assert parse_surface_marker_tag_id(uid=uid) == tag_id
        assert parse_surface_marker_tag_family(uid=uid) == tag_family


if __name__ == "__main__":
    test_surface_marker_from_raw_detection()
    test_surface_marker_deserialize()
    test_surface_marker_uid_helpers()
