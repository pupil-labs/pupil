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
    return ((SURFACE_V01_SQUARE_DESERIALIZED, SURFACE_V01_SQUARE_SERIALIZED),)


def surfaces_serialized() -> typing.Collection[dict]:
    return tuple(s for d, s in surface_pairs())


def surfaces_deserialized() -> typing.Collection[Surface]:
    return tuple(d for d, s in surface_pairs())


def surface_marker_aggregate_pairs() -> typing.Collection[
    typing.Tuple[Surface_Marker_Aggregate, dict]
]:
    return (
        (
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_0_DIST,
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_0_DIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_1_DIST,
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_1_DIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_0_UNDIST,
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_0_UNDIST,
        ),
        (
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_1_UNDIST,
            SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_1_UNDIST,
        ),
    )


def surface_marker_aggregates_serialized() -> typing.Collection[dict]:
    return tuple(s for d, s in surface_marker_aggregate_pairs())


def surface_marker_aggregates_deserialized() -> typing.Collection[
    Surface_Marker_Aggregate
]:
    return tuple(d for d, s in surface_marker_aggregate_pairs())


##### PRIVATE

SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_0_UNDIST = {
    "uid": "legacy:7",
    "verts_uv": [
        [1.4891731985457006e-14, -1.675372893802235e-14],
        [0.07250381261110306, -0.000125938662677072],
        [0.07127536088228226, 0.06990551203489304],
        [-0.00038871169090270996, 0.07084081321954727],
    ],
}
SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_1_UNDIST = {
    "uid": "legacy:57",
    "verts_uv": [
        [0.9311530590057373, 0.9295246005058289],
        [1.000424861907959, 0.9302592873573303],
        [1.0, 1.0],
        [0.9305340051651001, 0.9993915557861328],
    ],
}
SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_0_DIST = {
    "uid": "legacy:7",
    "verts_uv": [
        [1.1920822368235908e-14, -1.7095567718611662e-14],
        [0.06276015192270279, -0.0030732755549252033],
        [0.0531853586435318, 0.06535717844963074],
        [-0.010146145708858967, 0.06841480731964111],
    ],
}
SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_1_DIST = {
    "uid": "legacy:57",
    "verts_uv": [
        [0.9263750910758972, 0.9360179901123047],
        [0.9989929795265198, 0.9325454235076904],
        [1.0, 1.0],
        [0.9276587963104248, 1.0039196014404297],
    ],
}
SURFACE_V01_SQUARE_SERIALIZED = {
    "version": 1,
    "name": "square_surface_v01",
    "real_world_size": {"x": 1.0, "y": 1.0},
    "reg_markers": [
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_0_UNDIST,
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_1_UNDIST,
    ],
    "registered_markers_dist": [
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_0_DIST,
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_SERIALIZED_1_DIST,
    ],
    "build_up_status": 1.0,
    "deprecated": False,
}

SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_0_UNDIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:7"),
    verts_uv=np.asarray(
        [
            [1.4891731985457006e-14, -1.675372893802235e-14],
            [0.07250381261110306, -0.000125938662677072],
            [0.07127536088228226, 0.06990551203489304],
            [-0.00038871169090270996, 0.07084081321954727],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_1_UNDIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:57"),
    verts_uv=np.asarray(
        [
            [0.9311530590057373, 0.9295246005058289],
            [1.000424861907959, 0.9302592873573303],
            [1.0, 1.0],
            [0.9305340051651001, 0.9993915557861328],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_0_DIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:7"),
    verts_uv=np.asarray(
        [
            [1.1920822368235908e-14, -1.7095567718611662e-14],
            [0.06276015192270279, -0.0030732755549252033],
            [0.0531853586435318, 0.06535717844963074],
            [-0.010146145708858967, 0.06841480731964111],
        ]
    ),
)
SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_1_DIST = Surface_Marker_Aggregate(
    uid=Surface_Marker_UID("legacy:57"),
    verts_uv=np.asarray(
        [
            [0.9263750910758972, 0.9360179901123047],
            [0.9989929795265198, 0.9325454235076904],
            [1.0, 1.0],
            [0.9276587963104248, 1.0039196014404297],
        ]
    ),
)
SURFACE_V01_SQUARE_DESERIALIZED = Surface_Offline(
    name="square_surface_v01",
    real_world_size={"x": 1.0, "y": 1.0},
    marker_aggregates_undist=[
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_0_UNDIST,
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_1_UNDIST,
    ],
    marker_aggregates_dist=[
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_0_DIST,
        SURFACE_MARKER_AGGREGATE_V01_SQUARE_DESERIALIZED_1_DIST,
    ],
    build_up_status=1.0,
    deprecated_definition=False,
)
