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
    return ((SURFACE_V00_DESERIALIZED, SURFACE_V00_SERIALIZED),)


def surfaces_serialized() -> typing.Collection[dict]:
    return tuple(s for d, s in surface_pairs())


def surfaces_deserialized() -> typing.Collection[Surface]:
    return tuple(d for d, s in surface_pairs())


def surface_marker_aggregate_pairs() -> typing.Collection[
    typing.Tuple[Surface_Marker_Aggregate, dict]
]:
    return (
        (
            SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_0_DIST,
            SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_0_DIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_1_DIST,
            SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_1_DIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_0_UNDIST,
            SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_0_UNDIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_1_UNDIST,
            SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_1_UNDIST,
        ),
    )


def surface_marker_aggregates_serialized() -> typing.Collection[dict]:
    return tuple(s for d, s in surface_marker_aggregate_pairs())


def surface_marker_aggregates_deserialized() -> typing.Collection[
    Surface_Marker_Aggregate
]:
    return tuple(d for d, s in surface_marker_aggregate_pairs())


##### PRIVATE


SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_0_UNDIST = {
    "id": 7,
    "verts_uv": [
        [2.0279725503600556e-14, -2.0718593602363743e-14],
        [0.09232430905103683, 0.0054827057756483555],
        [0.09320462495088577, 0.07479614019393921],
        [0.008808332495391369, 0.07134716212749481],
    ],
}
SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_1_UNDIST = {
    "id": 57,
    "verts_uv": [
        [0.9255635738372803, 0.9278208017349243],
        [0.9941799640655518, 0.928483784198761],
        [0.9941900372505188, 0.9999602437019348],
        [0.9251440763473511, 0.998592734336853],
    ],
}
SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_0_DIST = {
    "id": 7,
    "verts_uv": [
        [1.9851928125457982e-14, -1.923472062778219e-14],
        [0.060702838003635406, -0.004638743586838245],
        [0.05217646434903145, 0.06511983275413513],
        [-0.009258653968572617, 0.06691507995128632],
    ],
}
SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_1_DIST = {
    "id": 57,
    "verts_uv": [
        [0.9114755799300843, 0.9409661768393776],
        [0.9818958957355025, 0.9379570537747127],
        [0.9800677671918846, 1.005555440640987],
        [0.909855488690773, 1.0082552654603305],
    ],
}
SURFACE_V00_SERIALIZED = {
    "name": "surface_v00",
    "real_world_size": {"x": 1.0, "y": 1.0},
    "reg_markers": [
        SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_0_UNDIST,
        SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_1_UNDIST,
    ],
    "registered_markers_dist": [
        SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_0_DIST,
        SURFACE_MARKER_AGGREGATE_V00_SERIALIZED_1_DIST,
    ],
    "build_up_status": 1.0,
    "deprecated": False,
}

SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_0_UNDIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:7"),
    verts_uv=np.asarray(
        [
            [2.0279725503600556e-14, -2.0718593602363743e-14],
            [0.09232430905103683, 0.0054827057756483555],
            [0.09320462495088577, 0.07479614019393921],
            [0.008808332495391369, 0.07134716212749481],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_1_UNDIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:57"),
    verts_uv=np.asarray(
        [
            [0.9255635738372803, 0.9278208017349243],
            [0.9941799640655518, 0.928483784198761],
            [0.9941900372505188, 0.9999602437019348],
            [0.9251440763473511, 0.998592734336853],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_0_DIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:7"),
    verts_uv=np.asarray(
        [
            [1.9851928125457982e-14, -1.923472062778219e-14],
            [0.060702838003635406, -0.004638743586838245],
            [0.05217646434903145, 0.06511983275413513],
            [-0.009258653968572617, 0.06691507995128632],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_1_DIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:57"),
    verts_uv=np.asarray(
        [
            [0.9114755799300843, 0.9409661768393776],
            [0.9818958957355025, 0.9379570537747127],
            [0.9800677671918846, 1.005555440640987],
            [0.909855488690773, 1.0082552654603305],
        ]
    ),
)
SURFACE_V00_DESERIALIZED = Surface_Offline(
    name="surface_v00",
    real_world_size={"x": 1.0, "y": 1.0},
    marker_aggregates_undist=[
        SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_0_UNDIST,
        SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_1_UNDIST,
    ],
    marker_aggregates_dist=[
        SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_0_DIST,
        SURFACE_MARKER_AGGREGATE_V00_DESERIALIZED_1_DIST,
    ],
    build_up_status=1.0,
    deprecated_definition=False,
)
