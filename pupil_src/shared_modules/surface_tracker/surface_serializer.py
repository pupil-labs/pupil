import abc
import functools
import logging
import os
import typing

from .surface import Surface
from .surface import Surface_Marker_Aggregate
from .surface_marker import Surface_Marker_UID, Surface_Marker_Type, Surface_Marker_TagID
from .surface_marker import create_surface_marker_uid, parse_surface_marker_tag_id, parse_surface_marker_type


logger = logging.getLogger(__name__)


class _Surface_Serializer_Base(abc.ABC):

    @property
    @abc.abstractmethod
    def version(self) -> int:
        pass

    @abc.abstractmethod
    def dict_from_surface_marker_aggregate(self, surface_marker_aggregate: Surface_Marker_Aggregate) -> dict:
        pass

    @abc.abstractmethod
    def surface_marker_aggregate_from_dict(self, surface_marker_aggregate_dict: dict) -> Surface_Marker_Aggregate:
        pass


class _Surface_Serializer_V00(_Surface_Serializer_Base):

    version = 0

    def dict_from_surface_marker_aggregate(self, surface_marker_aggregate: Surface_Marker_Aggregate) -> dict:
        id = parse_surface_marker_tag_id(uid=surface_marker_aggregate.uid)
        marker_type = parse_surface_marker_type(uid=surface_marker_aggregate.uid)
        if marker_type != Surface_Marker_Type.SQUARE:
            err_msg = f"{type(self).__name__} can only recognize {Surface_Marker_Type.SQUARE.value} markers"
            raise InvalidSurfaceDefinition(err_msg)
        verts_uv = surface_marker_aggregate.verts_uv
        if verts_uv is not None:
            verts_uv = [v.tolist() for v in verts_uv]
        return {"id": id, "verts_uv": verts_uv}

    def surface_marker_aggregate_from_dict(self, surface_marker_aggregate_dict: dict) -> Surface_Marker_Aggregate:
        tag_id = surface_marker_aggregate_dict["id"]
        uid = create_surface_marker_uid(
            marker_type=Surface_Marker_Type.SQUARE,
            tag_family=None,
            tag_id=Surface_Marker_TagID(tag_id)
        )
        verts_uv = surface_marker_aggregate_dict["verts_uv"]
        return Surface_Marker_Aggregate(uid=uid, verts_uv=verts_uv)


class _Surface_Serializer_V01(_Surface_Serializer_Base):

    version = 1

    def dict_from_surface_marker_aggregate(self, surface_marker_aggregate: Surface_Marker_Aggregate) -> dict:
        uid = str(surface_marker_aggregate.uid)
        verts_uv = surface_marker_aggregate.verts_uv
        if verts_uv is not None:
            verts_uv = [v.tolist() for v in verts_uv]
        return {"uid": uid, "verts_uv": verts_uv}

    def surface_marker_aggregate_from_dict(self, surface_marker_aggregate_dict: dict) -> Surface_Marker_Aggregate:
        uid = surface_marker_aggregate_dict["uid"]
        verts_uv = surface_marker_aggregate_dict["verts_uv"]
        return Surface_Marker_Aggregate(uid=uid, verts_uv=verts_uv)
