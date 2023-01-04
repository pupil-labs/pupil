"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import sys

from surface_tracker.surface import Surface, Surface_Marker_Aggregate
from surface_tracker.surface_serializer import (
    _Surface_Serializer_Base,
    _Surface_Serializer_V00,
    _Surface_Serializer_V01,
)

from .fixtures import (
    surface_marker_aggregate_pairs_v00_square,
    surface_marker_aggregate_pairs_v01_apriltag,
    surface_marker_aggregate_pairs_v01_square,
    surface_pairs_v00_square,
    surface_pairs_v01_apriltag,
    surface_pairs_v01_square,
)


def _test_surface_serializer_with_surfaces(serializer, surface_pairs):

    assert isinstance(serializer, _Surface_Serializer_Base)
    assert len(surface_pairs) > 0

    for deserialized_surface, serialized_surface in surface_pairs:
        assert isinstance(deserialized_surface, Surface)
        assert isinstance(serialized_surface, dict)

        serialization_result = serializer.dict_from_surface(
            surface=deserialized_surface
        )
        assert serialization_result == serialized_surface

        deserialization_result = serializer.surface_from_dict(
            surface_class=type(deserialized_surface),
            surface_definition=serialized_surface,
        )
        assert Surface.property_equality(deserialization_result, deserialized_surface)


def _test_surface_serializer_with_surface_marker_aggregates(
    serializer, aggregate_pairs
):

    assert isinstance(serializer, _Surface_Serializer_Base)
    assert len(aggregate_pairs) > 0

    for deserialized_aggregate, serialized_aggregate in aggregate_pairs:
        assert isinstance(deserialized_aggregate, Surface_Marker_Aggregate)
        assert isinstance(serialized_aggregate, dict)

        serialization_result = serializer.dict_from_surface_marker_aggregate(
            deserialized_aggregate
        )
        assert serialization_result == serialized_aggregate

        deserialization_result = serializer.surface_marker_aggregate_from_dict(
            serialized_aggregate
        )
        assert deserialization_result == deserialized_aggregate


def test_surface_serializer_V00():
    _test_surface_serializer_with_surfaces(
        serializer=_Surface_Serializer_V00(), surface_pairs=surface_pairs_v00_square()
    )
    _test_surface_serializer_with_surface_marker_aggregates(
        serializer=_Surface_Serializer_V00(),
        aggregate_pairs=surface_marker_aggregate_pairs_v00_square(),
    )


def test_surface_serializer_V01_square():
    _test_surface_serializer_with_surfaces(
        serializer=_Surface_Serializer_V01(), surface_pairs=surface_pairs_v01_square()
    )
    _test_surface_serializer_with_surface_marker_aggregates(
        serializer=_Surface_Serializer_V01(),
        aggregate_pairs=surface_marker_aggregate_pairs_v01_square(),
    )


def test_surface_serializer_V01_apriltag():
    _test_surface_serializer_with_surfaces(
        serializer=_Surface_Serializer_V01(), surface_pairs=surface_pairs_v01_apriltag()
    )
    _test_surface_serializer_with_surface_marker_aggregates(
        serializer=_Surface_Serializer_V01(),
        aggregate_pairs=surface_marker_aggregate_pairs_v01_apriltag(),
    )
