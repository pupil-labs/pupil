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
import functools
import logging
import os
import typing

from .surface import Surface, Surface_Marker_Aggregate
from .surface_marker import (
    Surface_Marker_TagID,
    Surface_Marker_Type,
    Surface_Marker_UID,
    create_surface_marker_uid,
    parse_surface_marker_tag_id,
    parse_surface_marker_type,
)

logger = logging.getLogger(__name__)


class InvalidSurfaceDefinition(Exception):
    pass


class _Surface_Serializer_Base(abc.ABC):
    @property
    @abc.abstractmethod
    def version(self) -> int:
        pass

    @abc.abstractmethod
    def dict_from_surface_marker_aggregate(
        self, surface_marker_aggregate: Surface_Marker_Aggregate
    ) -> dict:
        pass

    @abc.abstractmethod
    def surface_marker_aggregate_from_dict(
        self, surface_marker_aggregate_dict: dict
    ) -> Surface_Marker_Aggregate:
        pass

    def dict_from_surface(self, surface: Surface) -> dict:
        dict_from_marker_aggregate = self.dict_from_surface_marker_aggregate

        reg_markers = [
            dict_from_marker_aggregate(marker_aggregate)
            for marker_aggregate in surface._registered_markers_undist.values()  # TODO: Provide a public property for this
        ]
        registered_markers_dist = [
            dict_from_marker_aggregate(marker_aggregate)
            for marker_aggregate in surface._registered_markers_dist.values()  # TODO: Provide a public property for this
        ]
        return {
            "version": self.version,
            "name": surface.name,
            "real_world_size": surface.real_world_size,
            "reg_markers": reg_markers,
            "registered_markers_dist": registered_markers_dist,
            "build_up_status": surface.build_up_status,
            "deprecated": surface.deprecated_definition,
        }

    def surface_from_dict(self, surface_class, surface_definition: dict) -> Surface:
        assert isinstance(
            surface_class, type(object)
        ), f"surface_class must be a class: {surface_class}"
        assert issubclass(
            surface_class, Surface
        ), f"surface_class must be a subclass of Surface: {surface_class}"

        expected_version = self.version
        actual_version = surface_definition["version"]
        if actual_version != expected_version:
            err_msg = f"Invalid version, expected: {expected_version}, actual: {actual_version}"
            raise InvalidSurfaceDefinition(err_msg)

        marker_aggregate_from_dict = self.surface_marker_aggregate_from_dict

        marker_aggregates_undist = [
            marker_aggregate_from_dict(d) for d in surface_definition["reg_markers"]
        ]
        marker_aggregates_dist = [
            marker_aggregate_from_dict(d)
            for d in surface_definition["registered_markers_dist"]
        ]

        deprecated_definition = surface_definition.get("deprecated", True)

        if deprecated_definition:
            logger.warning(
                "You have loaded an old and deprecated surface definition! "
                "Please re-define this surface for increased mapping accuracy!"
            )

        return surface_class(
            name=surface_definition["name"],
            real_world_size=surface_definition["real_world_size"],
            marker_aggregates_undist=marker_aggregates_undist,
            marker_aggregates_dist=marker_aggregates_dist,
            build_up_status=surface_definition["build_up_status"],
            deprecated_definition=deprecated_definition,
        )


class _Surface_Serializer_V00(_Surface_Serializer_Base):

    version = 0

    def dict_from_surface_marker_aggregate(
        self, surface_marker_aggregate: Surface_Marker_Aggregate
    ) -> dict:
        id = parse_surface_marker_tag_id(uid=surface_marker_aggregate.uid)
        marker_type = parse_surface_marker_type(uid=surface_marker_aggregate.uid)
        if marker_type != Surface_Marker_Type.SQUARE:
            err_msg = f"{type(self).__name__} can only recognize {Surface_Marker_Type.SQUARE.value} markers"
            raise InvalidSurfaceDefinition(err_msg)
        verts_uv = surface_marker_aggregate.verts_uv
        if verts_uv is not None:
            verts_uv = [v.tolist() for v in verts_uv]
        return {"id": id, "verts_uv": verts_uv}

    def surface_marker_aggregate_from_dict(
        self, surface_marker_aggregate_dict: dict
    ) -> Surface_Marker_Aggregate:
        tag_id = surface_marker_aggregate_dict["id"]
        uid = create_surface_marker_uid(
            marker_type=Surface_Marker_Type.SQUARE,
            tag_family=None,
            tag_id=Surface_Marker_TagID(tag_id),
        )
        verts_uv = surface_marker_aggregate_dict["verts_uv"]
        return Surface_Marker_Aggregate(uid=uid, verts_uv=verts_uv)

    def dict_from_surface(self, surface: Surface) -> dict:
        surface_definition = super().dict_from_surface(surface=surface)
        # The format of v00 doesn't store any value for "version" key
        del surface_definition["version"]
        return surface_definition

    def surface_from_dict(self, surface_class, surface_definition: dict) -> Surface:
        # The format of v00 doesn't store any value for "version" key
        surface_definition["version"] = surface_definition.get("version", self.version)
        return super().surface_from_dict(
            surface_class=surface_class, surface_definition=surface_definition
        )


class _Surface_Serializer_V01(_Surface_Serializer_Base):

    version = 1

    def dict_from_surface_marker_aggregate(
        self, surface_marker_aggregate: Surface_Marker_Aggregate
    ) -> dict:
        uid = str(surface_marker_aggregate.uid)
        verts_uv = surface_marker_aggregate.verts_uv
        if verts_uv is not None:
            verts_uv = [v.tolist() for v in verts_uv]
        return {"uid": uid, "verts_uv": verts_uv}

    def surface_marker_aggregate_from_dict(
        self, surface_marker_aggregate_dict: dict
    ) -> Surface_Marker_Aggregate:
        uid = surface_marker_aggregate_dict["uid"]
        verts_uv = surface_marker_aggregate_dict["verts_uv"]
        return Surface_Marker_Aggregate(uid=uid, verts_uv=verts_uv)
