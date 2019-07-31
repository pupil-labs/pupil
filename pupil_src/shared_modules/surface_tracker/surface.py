"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import logging
import uuid

import cv2
import numpy as np

import methods
from surface_tracker.surface_marker_aggregate import Surface_Marker_Aggregate

logger = logging.getLogger(__name__)


class Surface(abc.ABC):
    """A Surface is a quadrangle whose position is defined in relation to a set of
    square markers in the real world. The markers are assumed to be in a fixed spatial
    relationship and to be in plane with one another as well as the surface."""

    def __init__(self, name="unknown", init_dict=None):
        self.name = name
        self.real_world_size = {"x": 1.0, "y": 1.0}
        self.deprecated_definition = False

        # We store the surface state in two versions: once computed with the
        # undistorted scene image and once with the still distorted scene image. The
        # undistorted state is used to map gaze onto the surface, the distorted one
        # is used for visualization. This is necessary because surface corners
        # outside of the image can not be re-distorted for visualization correctly.
        # Instead the slightly wrong but correct looking distorted version is
        # used for visualization.
        self.registered_markers_undist = {}
        self.registered_markers_dist = {}
        self.detected = False
        self.img_to_surf_trans = None
        self.surf_to_img_trans = None
        self.dist_img_to_surf_trans = None
        self.surf_to_dist_img_trans = None
        self.num_detected_markers = 0

        self._REQUIRED_OBS_PER_MARKER = 5
        self._avg_obs_per_marker = 0
        self.build_up_status = 0

        self.within_surface_heatmap = self.get_placeholder_heatmap()
        self.across_surface_heatmap = self.get_placeholder_heatmap()
        self._HEATMAP_MIN_DATA_CONFIDENCE = 0.6
        self._heatmap_scale = 0.5
        self._heatmap_resolution = 31
        self._heatmap_blur_factor = 0.0

        # The uid is only used to implement __hash__ and __eq__
        self._uid = uuid.uuid4()

        if init_dict is not None:
            self._load_from_dict(init_dict)

    def __hash__(self):
        return int(self._uid)

    def __eq__(self, other):
        if isinstance(other, Surface):
            return self._uid == other._uid
        else:
            return False

    @property
    def defined(self):
        return self.build_up_status >= 1.0

    def map_to_surf(
        self, points, camera_model, compensate_distortion=True, trans_matrix=None
    ):
        """Map points from image pixel space to normalized surface space.

        Args:
            points (ndarray): An array of 2D points with shape (2,) or (N, 2).
            camera_model: Camera Model object.
            compensate_distortion: If `True` camera distortion will be correctly
            compensated using the `camera_model`. Note that points that lie outside
            of the image can not be undistorted correctly and the attempt may
            introduce a large error.
            trans_matrix: The transformation matrix defining the location of
            the surface. If `None`, the current transformation matrix saved in the
            Surface object will be used.

        Returns:
            ndarray: Points mapped into normalized surface space. Has the same shape
            as the input.

        """
        if trans_matrix is None:
            if compensate_distortion:
                trans_matrix = self.img_to_surf_trans
            else:
                trans_matrix = self.dist_img_to_surf_trans

        if compensate_distortion:
            orig_shape = points.shape
            points = camera_model.undistort_points_on_image_plane(points)
            points.shape = orig_shape

        points_on_surf = self._perspective_transform_points(points, trans_matrix)

        return points_on_surf

    def map_from_surf(
        self, points, camera_model, compensate_distortion=True, trans_matrix=None
    ):
        """Map points from normalized surface space to image pixel space.

        Args:
            points (ndarray): An array of 2D points with shape (2,) or (N, 2).
            camera_model: Camera Model object.
            compensate_distortion: If `True` camera distortion will be correctly
            compensated using the `camera_model`. Note that points that lie outside
            of the image can not be undistorted correctly and the attempt may
            introduce a large error.
            trans_matrix: The transformation matrix defining the location of
            the surface. If `None`, the current transformation matrix saved in the
            Surface object will be used.

        Returns:
            ndarray: Points mapped into image space. Has the same shape
            as the input.

        """

        if trans_matrix is None:
            if compensate_distortion:
                trans_matrix = self.surf_to_img_trans
            else:
                trans_matrix = self.surf_to_dist_img_trans

        img_points = self._perspective_transform_points(points, trans_matrix)

        if compensate_distortion:
            orig_shape = points.shape
            img_points = camera_model.distortPoints(img_points)
            img_points.shape = orig_shape

        return img_points

    def _perspective_transform_points(self, points, trans_matrix):
        orig_shape = points.shape
        points.shape = (-1, 1, 2)
        img_points = cv2.perspectiveTransform(points, trans_matrix)
        img_points.shape = orig_shape
        return img_points

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
        """Computes a Surface_Location based on a list of visible markers."""

        visible_registered_marker_ids = set(visible_markers.keys()) & set(
            registered_markers_undist.keys()
        )

        # If the surface is defined by 2+ markers, we require 2+ markers to be detected.
        # If the surface is defined by 1 marker, we require 1 marker to be detected.
        if not visible_registered_marker_ids or len(
            visible_registered_marker_ids
        ) < min(2, len(registered_markers_undist)):
            return Surface_Location(detected=False)

        visible_verts_dist = np.array(
            [visible_markers[id].verts_px for id in visible_registered_marker_ids]
        )
        registered_verts_undist = np.array(
            [
                registered_markers_undist[id].verts_uv
                for id in visible_registered_marker_ids
            ]
        )
        registered_verts_dist = np.array(
            [
                registered_markers_dist[id].verts_uv
                for id in visible_registered_marker_ids
            ]
        )

        visible_verts_dist.shape = (-1, 2)
        registered_verts_undist.shape = (-1, 2)
        registered_verts_dist.shape = (-1, 2)

        dist_img_to_surf_trans, surf_to_dist_img_trans = Surface._find_homographies(
            registered_verts_dist, visible_verts_dist
        )

        visible_verts_undist = camera_model.undistort_points_on_image_plane(
            visible_verts_dist
        )
        img_to_surf_trans, surf_to_img_trans = Surface._find_homographies(
            registered_verts_undist, visible_verts_undist
        )

        return Surface_Location(
            True,
            dist_img_to_surf_trans,
            surf_to_dist_img_trans,
            img_to_surf_trans,
            surf_to_img_trans,
            len(visible_registered_marker_ids),
        )

    @staticmethod
    def _find_homographies(points_A, points_B):
        points_A = points_A.reshape((-1, 1, 2))
        points_B = points_B.reshape((-1, 1, 2))

        B_to_A, mask = cv2.findHomography(
            points_A, points_B, method=cv2.RANSAC, ransacReprojThreshold=100
        )

        A_to_B, mask = cv2.findHomography(points_B, points_A)
        return A_to_B, B_to_A

    def _update_definition(self, idx, visible_markers, camera_model):
        if not visible_markers:
            return

        all_verts_dist = np.array(
            [m.verts_px for m in visible_markers.values()], dtype=np.float32
        )
        all_verts_dist.shape = (-1, 2)
        all_verts_undist = camera_model.undistort_points_on_image_plane(all_verts_dist)

        hull_undist = self._bounding_quadrangle(all_verts_undist)
        undist_img_to_surf_trans_candidate = self._get_trans_to_norm_corners(
            hull_undist
        )
        all_verts_undist.shape = (-1, 1, 2)
        marker_surf_coords_undist = cv2.perspectiveTransform(
            all_verts_undist, undist_img_to_surf_trans_candidate
        )

        hull_dist = self._bounding_quadrangle(all_verts_dist)
        dist_img_to_surf_trans_candidate = self._get_trans_to_norm_corners(hull_dist)
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
                self.registered_markers_undist[marker.id].add_observation(uv_undist)
                self.registered_markers_dist[marker.id].add_observation(uv_dist)
            except KeyError:
                self.registered_markers_undist[marker.id] = Surface_Marker_Aggregate(
                    marker.id
                )
                self.registered_markers_undist[marker.id].add_observation(uv_undist)
                self.registered_markers_dist[marker.id] = Surface_Marker_Aggregate(
                    marker.id
                )
                self.registered_markers_dist[marker.id].add_observation(uv_dist)

        num_observations = sum(
            [len(m.observations) for m in self.registered_markers_undist.values()]
        )
        self._avg_obs_per_marker = num_observations / len(
            self.registered_markers_undist
        )
        self.build_up_status = self._avg_obs_per_marker / self._REQUIRED_OBS_PER_MARKER

        if self.build_up_status >= 1:
            self.prune_markers()

    def _bounding_quadrangle(self, vertices):

        # According to OpenCV implementation, cv2.convexHull only accepts arrays with
        # 32bit floats (CV_32F) or 32bit signed ints (CV_32S).
        # See: https://github.com/opencv/opencv/blob/3.4/modules/imgproc/src/convhull.cpp#L137
        # See: https://github.com/pupil-labs/pupil/issues/1544
        vertices = np.asarray(vertices, dtype=np.float32)

        hull_points = cv2.convexHull(vertices, clockwise=False)

        # The convex hull of a list of markers must have at least 4 corners, since a
        # single marker already has 4 corners. If the convex hull has more than 4
        # corners we reduce that number with approximations of the hull.
        if len(hull_points) > 4:
            new_hull = cv2.approxPolyDP(hull_points, epsilon=1, closed=True)
            if new_hull.shape[0] >= 4:
                hull_points = new_hull

        if len(hull_points) > 4:
            curvature = abs(methods.GetAnglesPolyline(hull_points, closed=True))
            most_acute_4_threshold = sorted(curvature)[3]
            hull_points = hull_points[curvature <= most_acute_4_threshold]

        # Vertices space is flipped in y.  We need to change the order of the
        # hull_points vertices
        hull_points = hull_points[[1, 0, 3, 2], :, :]

        # Roll the hull_points vertices until we have the right orientation:
        # vertices space has its origin at the image center. Adding 1 to the
        # coordinates puts the origin at the top left.
        distance_to_top_left = np.sqrt(
            (hull_points[:, :, 0] + 1) ** 2 + (hull_points[:, :, 1] + 1) ** 2
        )
        bot_left_idx = np.argmin(distance_to_top_left) + 1
        hull_points = np.roll(hull_points, -bot_left_idx, axis=0)
        return hull_points

    def _get_trans_to_norm_corners(self, verts):
        norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
        return cv2.getPerspectiveTransform(verts, norm_corners)

    def prune_markers(self):
        """Prune markers that are not support by sufficient observations."""
        persistent_markers = {}
        persistent_markers_dist = {}
        for (k, m), m_dist in zip(
            self.registered_markers_undist.items(),
            self.registered_markers_dist.values(),
        ):
            if len(m.observations) > self._REQUIRED_OBS_PER_MARKER * 0.5:
                persistent_markers[k] = m
                persistent_markers_dist[k] = m_dist
        self.registered_markers_undist = persistent_markers
        self.registered_markers_dist = persistent_markers_dist

    def move_corner(self, corner_idx, new_pos, camera_model):
        """Update surface definition by moving one of the corners to a new position.

        Args:
            corner_idx: Identifier of the corner to be moved. The order of corners is
            `[(0, 0), (1, 0), (1, 1), (0, 1)]`
            new_pos: The updated position of the corner in image pixel coordinates.
            camera_model: Camera Model object.

        """
        self._move_corner(
            corner_idx, new_pos, camera_model, self.registered_markers_undist, True
        )

        self._move_corner(
            corner_idx,
            new_pos,
            camera_model,
            markers=self.registered_markers_dist,
            compensate_distortion=False,
        )

    def _move_corner(
        self, corner_idx, new_pos, camera_model, markers, compensate_distortion
    ):
        # Markers undistorted
        new_corner_pos = self.map_to_surf(
            new_pos, camera_model, compensate_distortion=compensate_distortion
        )
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[corner_idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for marker in markers.values():
            marker.verts_uv = cv2.perspectiveTransform(
                marker.verts_uv.reshape((-1, 1, 2)), transform
            ).reshape((-1, 2))

    def add_marker(self, marker_id, verts_px, camera_model):
        self._add_marker(
            marker_id,
            verts_px,
            camera_model,
            markers=self.registered_markers_undist,
            compensate_distortion=True,
        )
        self._add_marker(
            marker_id,
            verts_px,
            camera_model,
            markers=self.registered_markers_dist,
            compensate_distortion=False,
        )

    def _add_marker(
        self, marker_id, verts_px, camera_model, markers, compensate_distortion
    ):
        surface_marker_dist = Surface_Marker_Aggregate(marker_id)
        marker_verts_dist = np.array(verts_px).reshape((4, 2))
        uv_coords_dist = self.map_to_surf(
            marker_verts_dist, camera_model, compensate_distortion=compensate_distortion
        )
        surface_marker_dist.add_observation(uv_coords_dist)
        markers[marker_id] = surface_marker_dist

    def pop_marker(self, id):
        self.registered_markers_dist.pop(id)
        self.registered_markers_undist.pop(id)

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
            self.within_surface_heatmap = self.get_uniform_heatmap()
            return

        color_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
        # reuse allocated memory if possible
        if self.within_surface_heatmap.shape != (*grid, 4):
            self.within_surface_heatmap = np.ones((*grid, 4), dtype=np.uint8)
            self.within_surface_heatmap[:, :, 3] = 125
        self.within_surface_heatmap[:, :, :3] = color_map

    def get_uniform_heatmap(self):
        hm = np.zeros((1, 1, 4), dtype=np.uint8)
        hm[:, :, :3] = cv2.applyColorMap(hm[:, :, :3], cv2.COLORMAP_JET)
        hm[:, :, 3] = 125
        return hm

    def get_placeholder_heatmap(self):
        hm = np.zeros((1, 1, 4), dtype=np.uint8)
        hm[:, :, 3] = 175

        return hm

    def save_to_dict(self):
        return {
            "name": self.name,
            "real_world_size": self.real_world_size,
            "reg_markers": [
                marker.save_to_dict()
                for marker in self.registered_markers_undist.values()
            ],
            "registered_markers_dist": [
                marker.save_to_dict()
                for marker in self.registered_markers_dist.values()
            ],
            "build_up_status": self.build_up_status,
            "deprecated": self.deprecated_definition,
        }

    def _load_from_dict(self, init_dict):
        self.name = init_dict["name"]
        self.real_world_size = init_dict["real_world_size"]
        self.registered_markers_undist = [
            Surface_Marker_Aggregate(marker["id"], verts_uv=marker["verts_uv"])
            for marker in init_dict["reg_markers"]
        ]
        self.registered_markers_undist = {
            m.id: m for m in self.registered_markers_undist
        }
        self.registered_markers_dist = [
            Surface_Marker_Aggregate(marker["id"], verts_uv=marker["verts_uv"])
            for marker in init_dict["registered_markers_dist"]
        ]
        self.registered_markers_dist = {m.id: m for m in self.registered_markers_dist}
        self.build_up_status = init_dict["build_up_status"]

        try:
            self.deprecated_definition = init_dict["deprecated"]
        except KeyError:
            pass
        else:
            logger.warning(
                "You have loaded an old and deprecated surface definition! "
                "Please re-define this surface for increased mapping accuracy!"
            )


class Surface_Location:
    def __init__(
        self,
        detected,
        dist_img_to_surf_trans=None,
        surf_to_dist_img_trans=None,
        img_to_surf_trans=None,
        surf_to_img_trans=None,
        num_detected_markers=0,
    ):
        self.detected = detected

        if self.detected:
            assert (
                dist_img_to_surf_trans is not None
                and surf_to_dist_img_trans is not None
                and img_to_surf_trans is not None
                and surf_to_img_trans is not None
                and num_detected_markers > 0
            ), (
                "Surface_Location can not be detected and have None as "
                "transformations at the same time!"
            )

        self.dist_img_to_surf_trans = dist_img_to_surf_trans
        self.surf_to_dist_img_trans = surf_to_dist_img_trans
        self.img_to_surf_trans = img_to_surf_trans
        self.surf_to_img_trans = surf_to_img_trans
        self.num_detected_markers = num_detected_markers

    def get_serializable_copy(self):
        location = {}
        location["detected"] = self.detected
        location["num_detected_markers"] = self.num_detected_markers
        if self.detected:
            location["dist_img_to_surf_trans"] = self.dist_img_to_surf_trans.tolist()
            location["surf_to_dist_img_trans"] = self.surf_to_dist_img_trans.tolist()
            location["img_to_surf_trans"] = self.img_to_surf_trans.tolist()
            location["surf_to_img_trans"] = self.surf_to_img_trans.tolist()
        else:
            location["dist_img_to_surf_trans"] = None
            location["surf_to_dist_img_trans"] = None
            location["img_to_surf_trans"] = None
            location["surf_to_img_trans"] = None
        return location

    @staticmethod
    def load_from_serializable_copy(copy):
        location = Surface_Location(detected=False)
        location.detected = copy["detected"]
        location.dist_img_to_surf_trans = np.asarray(copy["dist_img_to_surf_trans"])
        location.surf_to_dist_img_trans = np.asarray(copy["surf_to_dist_img_trans"])
        location.img_to_surf_trans = np.asarray(copy["img_to_surf_trans"])
        location.surf_to_img_trans = np.asarray(copy["surf_to_img_trans"])
        location.num_detected_markers = copy["num_detected_markers"]
        return location

    def __bool__(self):
        return self.detected
