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

from surface_tracker.surface import Surface
from surface_tracker.surface_marker_aggregate import Surface_Marker_Aggregate

from .fixtures_surface_v00_square import (
    surface_marker_aggregate_pairs as surface_marker_aggregate_pairs_v00_square,
)
from .fixtures_surface_v00_square import (
    surface_marker_aggregates_deserialized as surface_marker_aggregates_deserialized_v00_square,
)
from .fixtures_surface_v00_square import (
    surface_marker_aggregates_serialized as surface_marker_aggregates_serialized_v00_square,
)
from .fixtures_surface_v00_square import surface_pairs as surface_pairs_v00_square
from .fixtures_surface_v00_square import (
    surfaces_deserialized as surfaces_deserialized_v00_square,
)
from .fixtures_surface_v00_square import (
    surfaces_serialized as surfaces_serialized_v00_square,
)
from .fixtures_surface_v01_apriltag import (
    surface_marker_aggregate_pairs as surface_marker_aggregate_pairs_v01_apriltag,
)
from .fixtures_surface_v01_apriltag import (
    surface_marker_aggregates_deserialized as surface_marker_aggregates_deserialized_v01_apriltag,
)
from .fixtures_surface_v01_apriltag import (
    surface_marker_aggregates_serialized as surface_marker_aggregates_serialized_v01_apriltag,
)
from .fixtures_surface_v01_apriltag import surface_pairs as surface_pairs_v01_apriltag
from .fixtures_surface_v01_apriltag import (
    surfaces_deserialized as surfaces_deserialized_v01_apriltag,
)
from .fixtures_surface_v01_apriltag import (
    surfaces_serialized as surfaces_serialized_v01_apriltag,
)
from .fixtures_surface_v01_square import (
    surface_marker_aggregate_pairs as surface_marker_aggregate_pairs_v01_square,
)
from .fixtures_surface_v01_square import (
    surface_marker_aggregates_deserialized as surface_marker_aggregates_deserialized_v01_square,
)
from .fixtures_surface_v01_square import (
    surface_marker_aggregates_serialized as surface_marker_aggregates_serialized_v01_square,
)
from .fixtures_surface_v01_square import surface_pairs as surface_pairs_v01_square
from .fixtures_surface_v01_square import (
    surfaces_deserialized as surfaces_deserialized_v01_square,
)
from .fixtures_surface_v01_square import (
    surfaces_serialized as surfaces_serialized_v01_square,
)


def surface_pairs_v01_mixed() -> typing.Collection[typing.Tuple[Surface, dict]]:
    return (*surface_pairs_v01_apriltag(), *surface_pairs_v01_square())


def surfaces_serialized_v01_mixed() -> typing.Collection[dict]:
    return (*surfaces_serialized_v01_apriltag(), *surfaces_serialized_v01_square())


def surfaces_deserialized_v01_mixed() -> typing.Collection[Surface]:
    return (*surfaces_deserialized_v01_apriltag(), *surfaces_deserialized_v01_square())


def surface_marker_aggregate_pairs_v01_mixed() -> (
    typing.Collection[typing.Tuple[Surface_Marker_Aggregate, dict]]
):
    return (
        *surface_marker_aggregate_pairs_v01_apriltag(),
        *surface_marker_aggregate_pairs_v01_square(),
    )


def surface_marker_aggregates_serialized_v01_mixed() -> typing.Collection[dict]:
    return (
        *surface_marker_aggregates_serialized_v01_apriltag(),
        *surface_marker_aggregates_serialized_v01_square(),
    )


def surface_marker_aggregates_deserialized_v01_mixed() -> (
    typing.Collection[Surface_Marker_Aggregate]
):
    return (
        *surface_marker_aggregates_deserialized_v01_apriltag(),
        *surface_marker_aggregates_deserialized_v01_square(),
    )


from .fixtures_surface_definition_files import (
    surface_definition_v00_dir,
    surface_definition_v01_after_update_dir,
    surface_definition_v01_before_update_dir,
)
