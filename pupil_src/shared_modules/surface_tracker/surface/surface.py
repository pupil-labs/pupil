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
import typing
import logging
import operator
import uuid

import cv2
import numpy as np

import methods
from stdlib_utils import is_none, is_not_none

from surface_tracker.surface_marker import Surface_Marker_UID
from surface_tracker.surface_marker_aggregate import Surface_Marker_Aggregate
from .surface_location import Surface_Location
from . import surface_utils

logger = logging.getLogger(__name__)


Surface_Marker_UID_To_Aggregate_Mapping = typing.Mapping[
    Surface_Marker_UID, Surface_Marker_Aggregate
]


class Surface(abc.ABC):
    """A Surface is a quadrangle whose position is defined in relation to a set of
    square markers in the real world. The markers are assumed to be in a fixed spatial
    relationship and to be in plane with one another as well as the surface."""

    def __init__(
        self,
        name="unknown",
        real_world_size=None,
        marker_aggregates_undist=None,
        marker_aggregates_dist=None,
        build_up_status: float = None,
        deprecated_definition: bool = None,
    ):


        init_args = [
            real_world_size,
            marker_aggregates_undist,
            marker_aggregates_dist,
            build_up_status,
            deprecated_definition,
        ]
        assert all(map(is_none, init_args)) or all(map(is_not_none, init_args)),\
            "Either all initialization arguments are None, or they all are not None"

        marker_aggregates_undist = marker_aggregates_undist if marker_aggregates_undist is not None else []
        marker_aggregates_dist = marker_aggregates_dist if marker_aggregates_dist is not None else []

        self.name = name
        self.real_world_size = real_world_size if real_world_size is not None else {"x": 1.0, "y": 1.0}
        self.deprecated_definition = deprecated_definition if deprecated_definition is not None else False

        # We store the surface state in two versions: once computed with the
        # undistorted scene image and once with the still distorted scene image. The
        # undistorted state is used to map gaze onto the surface, the distorted one
        # is used for visualization. This is necessary because surface corners
        # outside of the image can not be re-distorted for visualization correctly.
        # Instead the slightly wrong but correct looking distorted version is
        # used for visualization.
        self._registered_markers_undist: Surface_Marker_UID_To_Aggregate_Mapping = {
            aggregate.uid: aggregate for aggregate in marker_aggregates_undist
        }
        self._registered_markers_dist: Surface_Marker_UID_To_Aggregate_Mapping = {
            aggregate.uid: aggregate for aggregate in marker_aggregates_dist
        }

        self.detected = False
        self.img_to_surf_trans = None
        self.surf_to_img_trans = None
        self.dist_img_to_surf_trans = None
        self.surf_to_dist_img_trans = None
        self.num_detected_markers = 0

        self._REQUIRED_OBS_PER_MARKER = 5
        self._avg_obs_per_marker = 0
        self.build_up_status = build_up_status if build_up_status is not None else 0

        self.within_surface_heatmap = surface_utils.placeholder_heatmap()
        self.across_surface_heatmap = surface_utils.placeholder_heatmap()
        self._HEATMAP_MIN_DATA_CONFIDENCE = 0.6
        self._heatmap_scale = 0.5
        self._heatmap_resolution = 31
        self._heatmap_blur_factor = 0.0

        # The uid is only used to implement __hash__ and __eq__
        self._uid = uuid.uuid4()



    def __hash__(self):
        return int(self._uid)

    def __eq__(self, other):
        if isinstance(other, Surface):
            return self._uid == other._uid
        else:
            return False

    @staticmethod
    def property_equality(x: "Surface", y: "Surface") -> bool:
        import multiprocessing as mp
        def property_dict(x: Surface) -> dict:
            x_dict = x.__dict__.copy()
            del x_dict["_uid"]  # `_uid`s are always unique
            for key in x_dict.keys():
                if isinstance(x_dict[key], np.ndarray):
                    x_dict[key] = x_dict[key].tolist()
                if isinstance(x_dict[key], mp.sharedctypes.Synchronized):
                    x_dict[key] = x_dict[key].value
            return x_dict
        return property_dict(x) == property_dict(y)

    @property
    def defined(self):
        return self.build_up_status >= 1.0

    @property
    def registered_markers_dist(self) -> Surface_Marker_UID_To_Aggregate_Mapping:
        return self._registered_markers_dist

    @property
    def registered_markers_undist(self) -> Surface_Marker_UID_To_Aggregate_Mapping:
        return self._registered_markers_undist

    def map_to_surf(
        self, points, camera_model, compensate_distortion=True, trans_matrix=None
    ):
        # TODO: Inline all method calls
        return surface_utils.map_to_surf(
            points=points,
            camera_model=camera_model,
            img_to_surf_trans=self.img_to_surf_trans,
            dist_img_to_surf_trans=self.dist_img_to_surf_trans,
            compensate_distortion=compensate_distortion,
            trans_matrix=trans_matrix
        )

    def map_from_surf(
        self, points, camera_model, compensate_distortion=True, trans_matrix=None
    ):
        # TODO: Inline all method calls
        return surface_utils.map_from_surf(
            points=points,
            camera_model=camera_model,
            surf_to_img_trans=self.surf_to_img_trans,
            surf_to_dist_img_trans=self.surf_to_dist_img_trans,
            compensate_distortion=compensate_distortion,
            trans_matrix=trans_matrix
        )

    def map_gaze_and_fixation_events(self, events, camera_model, trans_matrix=None):
        """
        Map a list of gaze or fixation events onto the surface and return the
        corresponding list of gaze/fixation on surface events.

        Args:
            events: List of gaze or fixation events.
            camera_model: Camera Model object.
            trans_matrix: The transformation matrix defining the location of
            the surface. If `None`, the current transformation matrix saved in the
            Surface object will be used.

        Returns:
            List of gaze or fixation on surface events.

        """
        results = []
        for event in events:
            gaze_norm_pos = event["norm_pos"]
            gaze_img_point = methods.denormalize(
                gaze_norm_pos, camera_model.resolution, flip_y=True
            )
            gaze_img_point = np.array(gaze_img_point)
            surf_norm_pos = self.map_to_surf(
                gaze_img_point,
                camera_model,
                compensate_distortion=True,
                trans_matrix=trans_matrix,
            )
            on_srf = bool((0 <= surf_norm_pos[0] <= 1) and (0 <= surf_norm_pos[1] <= 1))

            mapped_datum = {
                "topic": f"{event['topic']}_on_surface",
                "norm_pos": surf_norm_pos.tolist(),
                "confidence": event["confidence"],
                "on_surf": on_srf,
                "base_data": (event["topic"], event["timestamp"]),
                "timestamp": event["timestamp"],
            }
            if event["topic"] == "fixations":
                mapped_datum["id"] = event["id"]
                mapped_datum["duration"] = event["duration"]
                mapped_datum["dispersion"] = event["dispersion"]
            results.append(mapped_datum)
        return results

    @abc.abstractmethod
    def update_location(self, frame_idx, visible_markers, camera_model):
        """Update surface location based on marker detections in the current frame."""
        pass

    @staticmethod
    def locate(
        visible_markers,
        camera_model,
        registered_markers_undist,
        registered_markers_dist,
    ):
        # TODO: Inline all method calls
        return surface_utils.locate_surface(
            visible_markers,
            camera_model,
            registered_markers_undist,
            registered_markers_dist,
        )

    def _update_definition(self, idx, visible_markers, camera_model):
        if not visible_markers:
            return

        all_verts_dist = np.array(
            [m.verts_px for m in visible_markers.values()], dtype=np.float32
        )
        all_verts_dist.shape = (-1, 2)
        all_verts_undist = camera_model.undistort_points_on_image_plane(all_verts_dist)

        hull_undist = surface_utils.bounding_quadrangle(all_verts_undist)
        undist_img_to_surf_trans_candidate = surface_utils.perspective_transform_to_norm_corners(
            hull_undist
        )
        all_verts_undist.shape = (-1, 1, 2)
        marker_surf_coords_undist = cv2.perspectiveTransform(
            all_verts_undist, undist_img_to_surf_trans_candidate
        )

        hull_dist = surface_utils.bounding_quadrangle(all_verts_dist)
        dist_img_to_surf_trans_candidate = surface_utils.perspective_transform_to_norm_corners(hull_dist)
        all_verts_dist.shape = (-1, 1, 2)
        marker_surf_coords_dist = cv2.perspectiveTransform(
            all_verts_dist, dist_img_to_surf_trans_candidate
        )

        # Reshape to [marker, marker...]
        # where marker = [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
        marker_surf_coords_undist.shape = (-1, 4, 2)
        marker_surf_coords_dist.shape = (-1, 4, 2)

        # Add observations to library
        for marker, uv_undist, uv_dist in zip(
            visible_markers.values(), marker_surf_coords_undist, marker_surf_coords_dist
        ):
            try:
                self.registered_markers_undist[marker.uid].add_observation(uv_undist)
                self.registered_markers_dist[marker.uid].add_observation(uv_dist)
            except KeyError:
                self.registered_markers_undist[marker.uid] = Surface_Marker_Aggregate(
                    uid=marker.uid
                )
                self.registered_markers_undist[marker.uid].add_observation(uv_undist)
                self.registered_markers_dist[marker.uid] = Surface_Marker_Aggregate(
                    uid=marker.uid
                )
                self.registered_markers_dist[marker.uid].add_observation(uv_dist)

        num_observations = sum(
            [
                len(aggregate.observations)
                for aggregate in self.registered_markers_undist.values()
            ]
        )
        self._avg_obs_per_marker = num_observations / len(
            self.registered_markers_undist
        )
        self.build_up_status = self._avg_obs_per_marker / self._REQUIRED_OBS_PER_MARKER

        if self.build_up_status >= 1:
            self.prune_markers()

    def prune_markers(self):
        """Prune markers that are not support by sufficient observations."""
        persistent_markers = {}
        persistent_markers_dist = {}
        for (k, m), m_dist in zip(
            self.registered_markers_undist.items(),
            self._registered_markers_dist.values(),
        ):
            if len(m.observations) > self._REQUIRED_OBS_PER_MARKER * 0.5:
                persistent_markers[k] = m
                persistent_markers_dist[k] = m_dist
        self._registered_markers_undist = persistent_markers
        self._registered_markers_dist = persistent_markers_dist

    def move_corner(self, corner_idx, new_pos, camera_model):
        """Update surface definition by moving one of the corners to a new position.

        Args:
            corner_idx: Identifier of the corner to be moved. The order of corners is
            `[(0, 0), (1, 0), (1, 1), (0, 1)]`
            new_pos: The updated position of the corner in image pixel coordinates.
            camera_model: Camera Model object.

        """
        self._move_corner(
            corner_idx,
            new_pos,
            camera_model,
            marker_aggregate_mapping=self.registered_markers_undist,
            compensate_distortion=True,
        )

        self._move_corner(
            corner_idx,
            new_pos,
            camera_model,
            marker_aggregate_mapping=self._registered_markers_dist,
            compensate_distortion=False,
        )

    def _move_corner(
        self,
        corner_idx: int,
        new_pos,
        camera_model,
        marker_aggregate_mapping: Surface_Marker_UID_To_Aggregate_Mapping,
        compensate_distortion: bool,
    ):
        # Markers undistorted
        new_corner_pos = self.map_to_surf(
            new_pos, camera_model, compensate_distortion=compensate_distortion
        )
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[corner_idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for marker_aggregate in marker_aggregate_mapping.values():
            marker_aggregate.verts_uv = cv2.perspectiveTransform(
                marker_aggregate.verts_uv.reshape((-1, 1, 2)), transform
            ).reshape((-1, 2))

    def add_marker(self, marker_id, verts_px, camera_model):
        self._add_marker(
            marker_id,
            verts_px,
            camera_model,
            markers=self._registered_markers_undist,
            compensate_distortion=True,
        )
        self._add_marker(
            marker_id,
            verts_px,
            camera_model,
            markers=self._registered_markers_dist,
            compensate_distortion=False,
        )

    def _add_marker(
        self,
        marker_uid: Surface_Marker_UID,
        verts_px,
        camera_model,
        markers: Surface_Marker_UID_To_Aggregate_Mapping,
        compensate_distortion,
    ):
        surface_marker_dist = Surface_Marker_Aggregate(uid=marker_uid)
        marker_verts_dist = np.array(verts_px).reshape((4, 2))
        uv_coords_dist = self.map_to_surf(
            marker_verts_dist, camera_model, compensate_distortion=compensate_distortion
        )
        surface_marker_dist.add_observation(uv_coords_dist)
        markers[marker_uid] = surface_marker_dist

    def pop_marker(self, marker_uid: Surface_Marker_UID):
        self._registered_markers_dist.pop(marker_uid)
        self._registered_markers_undist.pop(marker_uid)

    def update_heatmap(self, gaze_on_surf):
        """Compute the gaze distribution heatmap based on given gaze events."""

        heatmap_data = [
            g["norm_pos"]
            for g in gaze_on_surf
            if g["on_surf"] and g["confidence"] >= self._HEATMAP_MIN_DATA_CONFIDENCE
        ]
        aspect_ratio = self.real_world_size["y"] / self.real_world_size["x"]
        grid = (
            max(1, int(self._heatmap_resolution * aspect_ratio)),
            int(self._heatmap_resolution),
        )
        if heatmap_data:
            xvals, yvals = zip(*((x, 1.0 - y) for x, y in heatmap_data))
            hist, *edges = np.histogram2d(
                yvals, xvals, bins=grid, range=[[0, 1.0], [0, 1.0]], normed=False
            )
            filter_h = 19 + self._heatmap_blur_factor * 15
            filter_w = filter_h * aspect_ratio
            filter_h = int(filter_h) // 2 * 2 + 1
            filter_w = int(filter_w) // 2 * 2 + 1

            hist = cv2.GaussianBlur(hist, (filter_h, filter_w), 0)
            hist_max = hist.max()
            hist *= (255.0 / hist_max) if hist_max else 0.0
            hist = hist.astype(np.uint8)
        else:
            self.within_surface_heatmap = surface_utils.uniform_heatmap(grid)
            return

        color_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
        # reuse allocated memory if possible
        if self.within_surface_heatmap.shape != (*grid, 4):
            self.within_surface_heatmap = np.ones((*grid, 4), dtype=np.uint8)
            self.within_surface_heatmap[:, :, 3] = 125
        self.within_surface_heatmap[:, :, :3] = color_map
