"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np
import cv2
from abc import ABCMeta, abstractmethod
import uuid

import methods

from surface_tracker import _Surface_Marker_Aggregate


class Surface(metaclass=ABCMeta):
    """A Surface is a quadrangel whose position is defined in relation to a set of
    square markers in the real world. The markers are assumed to be in a fixed spatial
    relationship and to be in plane with one another as well as the surface."""

    def __init__(self, name="unknown", init_dict=None):
        self.name = name
        self.real_world_size = {"x": 1., "y": 1.}

        # We store the surface state in two versions: once computed with the
        # undistorted scene image and once with the still distorted scene image. The
        # undistorted state is used to map gaze onto the surface, the distorted one
        # is used for visualization. This is necessary because surface corners
        # outside of the image can not be re-distorted for visualization correctly.
        # Instead the slightly wrong but correct looking distorted version is
        # used for visualization.
        self.reg_markers_undist = {}
        self.reg_markers_dist = {}
        self.detected = False
        self.img_to_surf_trans = None
        self.surf_to_img_trans = None
        self.dist_img_to_surf_trans = None
        self.surf_to_dist_img_trans = None

        self._REQUIRED_OBS_PER_MARKER = 5
        self._avg_obs_per_marker = 0
        self.build_up_status = 0

        self.within_surface_heatmap = self._get_dummy_heatmap()
        self.across_surface_heatmap = self._get_dummy_heatmap()
        self._HEATMAP_MIN_DATA_CONFIDENCE = 0.6
        self._heatmap_scale = 0.5
        self._heatmap_resolution = 31
        self._heatmap_blur_factor = 0.

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
            points = camera_model.undistortPoints(points)
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

            results.append(
                {
                    "topic": event["topic"] + "on_surface",
                    "norm_pos": surf_norm_pos.tolist(),
                    "confidence": event["confidence"],
                    "on_surf": on_srf,
                    "base_data": (event["topic"], event["timestamp"]),
                    "timestamp": event["timestamp"],
                }
            )
        return results

    @abstractmethod
    def update_location(self, frame_idx, visible_markers, camera_model):
        """Update surface location based on marker detections in the current frame."""
        pass

    @staticmethod
    def locate(visible_markers, camera_model, reg_markers_undist, reg_markers_dist):
        """Computes homographys mapping the surface from and to image pixel space."""

        result = {
            "detected": False,
            "dist_img_to_surf_trans": None,
            "surf_to_dist_img_trans": None,
            "img_to_surf_trans": None,
            "surf_to_img_trans": None,
            "num_detected_markers": 0,
        }

        vis_reg_marker_ids = set(visible_markers.keys()) & set(
            reg_markers_undist.keys()
        )

        if not vis_reg_marker_ids or len(vis_reg_marker_ids) < min(
            2, len(reg_markers_undist)
        ):
            return result

        vis_verts_dist = np.array(
            [visible_markers[id].verts_px for id in vis_reg_marker_ids]
        )
        reg_verts_undist = np.array(
            [reg_markers_undist[id].verts_uv for id in vis_reg_marker_ids]
        )
        reg_verts_dist = np.array(
            [reg_markers_dist[id].verts_uv for id in vis_reg_marker_ids]
        )

        vis_verts_dist.shape = (-1, 2)
        reg_verts_undist.shape = (-1, 2)
        reg_verts_dist.shape = (-1, 2)

        dist_img_to_surf_trans, surf_to_dist_img_trans = Surface._find_homographies(
            reg_verts_dist, vis_verts_dist
        )

        vis_verts_undist = camera_model.undistortPoints(vis_verts_dist)
        img_to_surf_trans, surf_to_img_trans = Surface._find_homographies(
            reg_verts_undist, vis_verts_undist
        )

        if img_to_surf_trans is None or dist_img_to_surf_trans is None:
            return result
        else:
            result["detected"] = True
            result["dist_img_to_surf_trans"] = dist_img_to_surf_trans
            result["surf_to_dist_img_trans"] = surf_to_dist_img_trans
            result["img_to_surf_trans"] = img_to_surf_trans
            result["surf_to_img_trans"] = surf_to_img_trans
            result["num_detected_markers"] = len(vis_reg_marker_ids)
            return result

    @staticmethod
    def _find_homographies(points_A, points_B):
        points_A = points_A.reshape((-1, 1, 2))
        points_B = points_B.reshape((-1, 1, 2))

        B_to_A, mask = cv2.findHomography(
            points_A, points_B, method=cv2.RANSAC, ransacReprojThreshold=100
        )

        if not mask.all():
            return None, None

        A_to_B, mask = cv2.findHomography(points_B, points_A)
        return A_to_B, B_to_A

    def _update_definition(self, idx, visible_markers, camera_model):
        if not visible_markers:
            return

        all_verts_dist = np.array(
            [m.verts_px for m in visible_markers.values()], dtype=np.float32
        )
        all_verts_dist.shape = (-1, 2)
        all_verts_undist = camera_model.undistortPoints(all_verts_dist)

        hull_undist = self._bounding_quadrangle(all_verts_undist)
        undist_img_to_surf_trans_candidate = self._get_trans_to_norm(hull_undist)
        all_verts_undist.shape = (-1, 1, 2)
        marker_surf_coords_undist = cv2.perspectiveTransform(
            all_verts_undist, undist_img_to_surf_trans_candidate
        )

        hull_dist = self._bounding_quadrangle(all_verts_dist)
        dist_img_to_surf_trans_candidate = self._get_trans_to_norm(hull_dist)
        all_verts_dist.shape = (-1, 1, 2)
        marker_surf_coords_dist = cv2.perspectiveTransform(
            all_verts_dist, dist_img_to_surf_trans_candidate
        )

        # Reshape to [marker, marker...]
        # where marker = [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
        marker_surf_coords_undist.shape = (-1, 4, 2)
        marker_surf_coords_dist.shape = (-1, 4, 2)

        # Add observations to library
        for m, uv_undist, uv_dist in zip(
            visible_markers.values(), marker_surf_coords_undist, marker_surf_coords_dist
        ):
            try:
                self.reg_markers_undist[m.id].add_observation(uv_undist)
                self.reg_markers_dist[m.id].add_observation(uv_dist)
            except KeyError:
                self.reg_markers_undist[m.id] = _Surface_Marker_Aggregate(m.id)
                self.reg_markers_undist[m.id].add_observation(uv_undist)
                self.reg_markers_dist[m.id] = _Surface_Marker_Aggregate(m.id)
                self.reg_markers_dist[m.id].add_observation(uv_dist)

        num_observations = sum(
            [len(m.observations) for m in self.reg_markers_undist.values()]
        )
        self._avg_obs_per_marker = num_observations / len(self.reg_markers_undist)
        self.build_up_status = self._avg_obs_per_marker / self._REQUIRED_OBS_PER_MARKER

        if self.build_up_status >= 1:
            self._finalize_def()

    def _bounding_quadrangle(self, verts):
        hull = cv2.convexHull(verts, clockwise=False)

        if hull.shape[0] > 4:
            new_hull = cv2.approxPolyDP(hull, epsilon=1, closed=True)
            if new_hull.shape[0] >= 4:
                hull = new_hull

        if hull.shape[0] > 4:
            curvature = abs(methods.GetAnglesPolyline(hull, closed=True))
            most_acute_4_threshold = sorted(curvature)[3]
            hull = hull[curvature <= most_acute_4_threshold]

        # verts space is flipped in y.
        # we need to change the order of the hull vertecies
        hull = hull[[1, 0, 3, 2], :, :]

        # now we need to roll the hull verts until we have the right orientation:
        # verts space has its origin at the image center.
        # adding 1 to the coordinates puts the origin at the top left.
        distance_to_top_left = np.sqrt(
            (hull[:, :, 0] + 1) ** 2 + (hull[:, :, 1] + 1) ** 2
        )
        bot_left_idx = np.argmin(distance_to_top_left) + 1
        hull = np.roll(hull, -bot_left_idx, axis=0)
        return hull

    def _get_trans_to_norm(self, verts):
        norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
        return cv2.getPerspectiveTransform(verts, norm_corners)

    def _denormalize(self, points, camera_model):
        if len(points.shape) == 1:
            points[1] = 1 - points[1]
        else:
            points[:, 1] = 1 - points[:, 1]
        points *= camera_model.resolution
        return points

    def _finalize_def(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        persistent_markers_dist = {}
        for (k, m), m_dist in zip(
            self.reg_markers_undist.items(), self.reg_markers_dist.values()
        ):
            if len(m.observations) > self._REQUIRED_OBS_PER_MARKER * .5:
                persistent_markers[k] = m
                persistent_markers_dist[k] = m_dist
        self.reg_markers_undist = persistent_markers
        self.reg_markers_dist = persistent_markers_dist

    def move_corner(self, corner_idx, new_pos, camera_model):
        """Update surface definition by moving one of the corners to a new position.

        Args:
            corner_idx: Identifier of the corner to be moved. The order of corners is
            `[(0, 0), (1, 0), (1, 1), (0, 1)]`
            new_pos: The updated position of the corner in image pixel coordinates.
            camera_model: Camera Model object.

        """
        # Markers undistorted
        new_corner_pos = self.map_to_surf(
            new_pos, camera_model, compensate_distortion=True
        )
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[corner_idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for m in self.reg_markers_undist.values():
            m.verts_uv = cv2.perspectiveTransform(
                m.verts_uv.reshape((-1, 1, 2)), transform
            ).reshape((-1, 2))

        # Markers distorted
        new_corner_pos = self.map_to_surf(
            new_pos, camera_model, compensate_distortion=False
        )
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[corner_idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for m in self.reg_markers_dist.values():
            m.verts_uv = cv2.perspectiveTransform(
                m.verts_uv.reshape((-1, 1, 2)), transform
            ).reshape((-1, 2))

    def add_marker(self, marker_id, verts_px, camera_model):
        """Add a marker to the surface definition."""
        surface_marker_dist = _Surface_Marker_Aggregate(marker_id)
        marker_verts_dist = np.array(verts_px).reshape((4, 2))
        uv_coords_dist = self.map_to_surf(
            marker_verts_dist, camera_model, compensate_distortion=False
        )
        surface_marker_dist.add_observation(uv_coords_dist)
        self.reg_markers_dist[marker_id] = surface_marker_dist

        surface_marker_undist = _Surface_Marker_Aggregate(marker_id)
        marker_verts_undist = np.array(verts_px).reshape((4, 2))
        uv_coords_undist = self.map_to_surf(
            marker_verts_undist, camera_model, compensate_distortion=False
        )
        surface_marker_undist.add_observation(uv_coords_undist)
        self.reg_markers_undist[marker_id] = surface_marker_undist

    def pop_marker(self, id):
        """Remove a marker from the surface definition."""
        self.reg_markers_dist.pop(id)
        self.reg_markers_undist.pop(id)

    def update_heatmap(self, gaze_on_surf):
        """Compute the gaze distribution heatmap based on given gaze events."""

        heatmap_data = [
            g["norm_pos"]
            for g in gaze_on_surf
            if g["on_surf"] and g["confidence"] >= self._HEATMAP_MIN_DATA_CONFIDENCE
        ]
        aspect_ratio = self.real_world_size["y"] / self.real_world_size["x"]
        grid = (
            int(self._heatmap_resolution),
            int(self._heatmap_resolution * aspect_ratio),
        )
        if heatmap_data:
            xvals, yvals = zip(*((x, 1. - y) for x, y in heatmap_data))
            hist, *edges = np.histogram2d(
                yvals, xvals, bins=grid, range=[[0, 1.], [0, 1.]], normed=False
            )
            filter_h = 19 + self._heatmap_blur_factor * 15
            filter_w = filter_h * aspect_ratio
            filter_h = int(filter_h) // 2 * 2 + 1
            filter_w = int(filter_w) // 2 * 2 + 1

            hist = cv2.GaussianBlur(hist, (filter_h, filter_w), 0)
            hist_max = hist.max()
            hist *= (255. / hist_max) if hist_max else 0.
            hist = hist.astype(np.uint8)
        else:
            self.within_surface_heatmap = self._get_dummy_heatmap(empty=True)
            return

        c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
        # reuse allocated memory if possible
        if self.within_surface_heatmap.shape != (*grid, 4):
            self.within_surface_heatmap = np.ones((*grid, 4), dtype=np.uint8)
            self.within_surface_heatmap[:, :, 3] = 125
        self.within_surface_heatmap[:, :, :3] = c_map

    def _get_dummy_heatmap(self, empty=False):
        hm = np.zeros((1, 1, 4), dtype=np.uint8)

        if empty:
            hm[:, :, :3] = cv2.applyColorMap(hm[:, :, :3], cv2.COLORMAP_JET)
            hm[:, :, 3] = 125
        else:
            hm[:, :, 3] = 175

        return hm

    def save_to_dict(self):
        return {
            "name": self.name,
            "real_world_size": self.real_world_size,
            "reg_markers": [
                marker.save_to_dict() for marker in self.reg_markers_undist.values()
            ],
            "reg_markers_dist": [
                marker.save_to_dict() for marker in self.reg_markers_dist.values()
            ],
            "build_up_status": self.build_up_status,
        }

    def _load_from_dict(self, init_dict):
        self.name = init_dict["name"]
        self.real_world_size = init_dict["real_world_size"]
        self.reg_markers_undist = [
            _Surface_Marker_Aggregate(marker["id"], verts_uv=marker["verts_uv"])
            for marker in init_dict["reg_markers"]
        ]
        self.reg_markers_undist = {m.id: m for m in self.reg_markers_undist}
        self.reg_markers_dist = [
            _Surface_Marker_Aggregate(marker["id"], verts_uv=marker["verts_uv"])
            for marker in init_dict["reg_markers_dist"]
        ]
        self.reg_markers_dist = {m.id: m for m in self.reg_markers_dist}
        self.build_up_status = init_dict["build_up_status"]
