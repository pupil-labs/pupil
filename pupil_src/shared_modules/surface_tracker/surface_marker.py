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
import collections
import enum
import functools
import itertools
import typing

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
    uid: Surface_Marker_UID,
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
        detection: Apriltag_V3_Detection,
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
