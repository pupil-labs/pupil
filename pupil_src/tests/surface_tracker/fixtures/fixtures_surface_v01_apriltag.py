"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing

import numpy as np
from surface_tracker.surface import Surface
from surface_tracker.surface_marker_aggregate import (
    Surface_Marker_Aggregate,
    Surface_Marker_UID,
)
from surface_tracker.surface_offline import Surface_Offline
from surface_tracker.surface_online import Surface_Online

__all__ = [
    "surface_pairs",
    "surfaces_serialized",
    "surfaces_deserialized",
    "surface_marker_aggregate_pairs",
    "surface_marker_aggregates_serialized",
    "surface_marker_aggregates_deserialized",
]


def surface_pairs() -> typing.Collection[typing.Tuple[Surface, dict]]:
    return ((SURFACE_V01_APRILTAG_DESERIALIZED, SURFACE_V01_APRILTAG_SERIALIZED),)


def surfaces_serialized() -> typing.Collection[dict]:
    return tuple(s for d, s in surface_pairs())


def surfaces_deserialized() -> typing.Collection[Surface]:
    return tuple(d for d, s in surface_pairs())


def surface_marker_aggregate_pairs() -> typing.Collection[
    typing.Tuple[Surface_Marker_Aggregate, dict]
]:
    return (
        (
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_0_DIST,
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_0_DIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_1_DIST,
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_1_DIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_0_UNDIST,
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_0_UNDIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_1_UNDIST,
            SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_1_UNDIST,
        ),
    )


def surface_marker_aggregates_serialized() -> typing.Collection[dict]:
    return tuple(s for d, s in surface_marker_aggregate_pairs())


def surface_marker_aggregates_deserialized() -> typing.Collection[
    Surface_Marker_Aggregate
]:
    return tuple(d for d, s in surface_marker_aggregate_pairs())


##### PRIVATE


SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_0_UNDIST = {
    "uid": "apriltag_v3:tag36h11:72",
    "verts_uv": [
        [0.9084336161613464, 0.9975847005844116],
        [0.908658504486084, 0.8936676979064941],
        [1.000919222831726, 0.8954850435256958],
        [1.0, 1.0],
    ],
}
SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_1_UNDIST = {
    "uid": "apriltag_v3:tag36h11:8",
    "verts_uv": [
        [0.008291917853057384, 0.12046175450086594],
        [0.0049503217451274395, 0.016743427142500877],
        [0.0993560180068016, 0.01397931668907404],
        [0.09943299740552902, 0.11729079484939575],
    ],
}
SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_0_DIST = {
    "uid": "apriltag_v3:tag36h11:72",
    "verts_uv": [
        [0.9030435085296631, 1.0059881210327148],
        [0.9017457962036133, 0.8991504907608032],
        [0.9998528361320496, 0.8934420347213745],
        [1.0, 1.0],
    ],
}
SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_1_DIST = {
    "uid": "apriltag_v3:tag36h11:8",
    "verts_uv": [
        [-0.01220057875846362, 0.11509560622293023],
        [0.0014339334968873486, 0.015284867423122411],
        [0.08342891010320912, 0.0064126273309583],
        [0.07023519482627806, 0.10786793886342823],
    ],
}
SURFACE_V01_APRILTAG_SERIALIZED = {
    "version": 1,
    "name": "apriltag_surface_v01",
    "real_world_size": {"x": 1.0, "y": 1.0},
    "reg_markers": [
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_0_UNDIST,
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_1_UNDIST,
    ],
    "registered_markers_dist": [
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_0_DIST,
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_SERIALIZED_1_DIST,
    ],
    "build_up_status": 1.0,
    "deprecated": False,
}

SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_0_UNDIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("apriltag_v3:tag36h11:72"),
    verts_uv=np.asarray(
        [
            [0.9084336161613464, 0.9975847005844116],
            [0.908658504486084, 0.8936676979064941],
            [1.000919222831726, 0.8954850435256958],
            [1.0, 1.0],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_1_UNDIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("apriltag_v3:tag36h11:8"),
    verts_uv=np.asarray(
        [
            [0.008291917853057384, 0.12046175450086594],
            [0.0049503217451274395, 0.016743427142500877],
            [0.0993560180068016, 0.01397931668907404],
            [0.09943299740552902, 0.11729079484939575],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_0_DIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("apriltag_v3:tag36h11:72"),
    verts_uv=np.asarray(
        [
            [0.9030435085296631, 1.0059881210327148],
            [0.9017457962036133, 0.8991504907608032],
            [0.9998528361320496, 0.8934420347213745],
            [1.0, 1.0],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_1_DIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("apriltag_v3:tag36h11:8"),
    verts_uv=np.asarray(
        [
            [-0.01220057875846362, 0.11509560622293023],
            [0.0014339334968873486, 0.015284867423122411],
            [0.08342891010320912, 0.0064126273309583],
            [0.07023519482627806, 0.10786793886342823],
        ]
    ),
)
SURFACE_V01_APRILTAG_DESERIALIZED = Surface_Offline(
    name="apriltag_surface_v01",
    real_world_size={"x": 1.0, "y": 1.0},
    marker_aggregates_undist=[
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_0_UNDIST,
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_1_UNDIST,
    ],
    marker_aggregates_dist=[
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_0_DIST,
        SURFACE_MARKER_AGGREGATE_V01_APRILTAG_DESERIALIZED_1_DIST,
    ],
    build_up_status=1.0,
    deprecated_definition=False,
)
