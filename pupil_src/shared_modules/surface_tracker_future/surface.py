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
import random
import collections

import methods


class Surface:
    def __init__(self, camera_model, marker_min_perimeter, marker_min_confidence,
                 init_dict=None):
        self.uid = random.randint(0, 1e6)

        self.camera_model = camera_model
        self.marker_min_perimeter = marker_min_perimeter
        self.marker_min_confidence = marker_min_confidence
        self.name = "unknown"
        self.real_world_size = {"x": 1., "y": 1.}

        self.reg_markers = {}
        self.img_to_surf_trans = None # TODO Do these need to be public?
        self.surf_to_img_trans = None
        self._dist_img_to_surf_trans = None
        self._surf_to_dist_img_trans = None

        self.detected = False
        self.num_detected_markers = []

        self.defined = False
        self._required_obs_per_marker = 30
        self._avg_obs_per_marker = 0
        self.build_up_status = 0

        self.gaze_history_length = 1
        self.gaze_history = collections.deque()
        self.heatmap = np.ones((1,1), dtype=np.uint8)
        self.heatmap_detail = .2

        if not init_dict is None:
            self.load_from_dict(init_dict)

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        if isinstance(Surface, other):
            return self.uid == other.uid
        else:
            return False

    def map_to_surf(self, points):
        """Map points from image space to normalized surface space.

        Args:
            points (ndarray): An array of points in one of the following shapes: (2,
            ), (N, 2), (N, 1, 2)

        Returns:
            ndarray: Points mapped into normalized surface space. Has the same shape
            as the input.

        """
        orig_shape = points.shape
        points = self.camera_model.undistortPoints(points)
        points.shape = (-1, 1, 2)
        point_on_surf = cv2.perspectiveTransform(points, self.img_to_surf_trans)
        point_on_surf.shape = orig_shape
        return point_on_surf

    def map_from_surf(self, points):
        """Map points from normalized surface space to image space.

        Args:
            points (ndarray): An array of points in one of the following shapes: (2,
            ), (N, 2), (N, 1, 2)

        Returns:
            ndarray: Points mapped into image space. Has the same shape
            as the input.

        """
        orig_shape = points.shape
        points.shape = (-1, 1, 2)
        img_points = cv2.perspectiveTransform(points, self.surf_to_img_trans)
        img_points = self.camera_model.distortPoints(img_points)
        img_points.shape = orig_shape
        return img_points

    def move_corner(self, idx, pos):
        new_corner_pos = self.map_to_surf(pos)
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for m in self.reg_markers.values():
            m.verts = cv2.perspectiveTransform(m.verts.reshape((-1, 1, 2)),
                                               transform).reshape((-1, 2))

    def update(self, vis_markers):
        vis_markers = self._filter_markers(vis_markers)

        if not self.defined:
            self._add_observation(vis_markers)

        self._update_location(vis_markers)

    def _add_observation(self, vis_markers):
        if not vis_markers:
            return

        all_verts = np.array([m.verts for m in vis_markers.values()], dtype=np.float32)
        all_verts.shape = (-1, 2)
        all_verts_undist = self.camera_model.undistortPoints(all_verts)
        hull = self._bounding_quadrangle(all_verts_undist)

        img_to_surf_trans_candidate = self._get_trans_to_norm(hull)
        all_verts_undist.shape = (-1, 1, 2)
        marker_surf_coords = cv2.perspectiveTransform(
            all_verts_undist, img_to_surf_trans_candidate
        )

        # Reshape to [marker, marker...]
        # where marker = [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
        marker_surf_coords.shape = (-1, 4, 2)

        # Add observations to library
        for m, uv in zip(vis_markers.values(), marker_surf_coords):
            try:
                self.reg_markers[m.id].add_observation(uv)
            except KeyError:
                self.reg_markers[m.id] = _Surface_Marker(m.id)
                self.reg_markers[m.id].add_observation(uv)

        num_observations = sum(
            [len(m.observations) for m in self.reg_markers.values()]
        )
        self._avg_obs_per_marker = num_observations / len(self.reg_markers)
        self.build_up_status = self._avg_obs_per_marker / self._required_obs_per_marker

        if self.build_up_status >= 1:
            self._finalize_surf_def()

    def _finalize_surf_def(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        for k, m in self.reg_markers.items():
            if len(m.observations) > self._required_obs_per_marker * .5:
                persistent_markers[k] = m
        self.reg_markers = persistent_markers
        self.defined = True

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

    def _update_location(self, vis_markers):
        vis_reg_marker_ids = set(vis_markers.keys()) & set(self.reg_markers.keys())
        self.num_detected_markers = len(vis_reg_marker_ids)

        if not vis_reg_marker_ids or len(vis_reg_marker_ids) < min(
            2, len(self.reg_markers)
        ):
            self.detected = False
            self.img_to_surf_trans = None
            self.surf_to_img_trans = None
            return

        vis_verts = np.array([vis_markers[id].verts for id in vis_reg_marker_ids])
        reg_verts = np.array(
            [self.reg_markers[id].verts for id in vis_reg_marker_ids]
        )

        vis_verts.shape = (-1, 2)
        reg_verts.shape = (-1, 2)

        vis_verts_undist = self.camera_model.undistortPoints(vis_verts)
        self.img_to_surf_trans, self.surf_to_img_trans = self._findHomographies(
            reg_verts, vis_verts_undist
        )
        self._dist_img_to_surf_trans, self._surf_to_dist_img_trans = self._findHomographies(
            reg_verts, vis_verts
        )

        if self.img_to_surf_trans is None or self._dist_img_to_surf_trans is None:
            self.img_to_surf_trans = None  # TODO Do these need to be public?
            self.surf_to_img_trans = None
            self._dist_img_to_surf_trans = None
            self._surf_to_dist_img_trans = None
            self.detected = False
        else:
            self.detected = True

    def _filter_markers(self, visible_markers):
        filtered_markers = [
            m
            for m in visible_markers
            if m.perimeter >= self.marker_min_perimeter
            and m.id_confidence > self.marker_min_confidence
        ]

        # if an id shows twice use the bigger marker (usually this is a screen camera echo artifact.)
        marker_by_id = {}
        for m in filtered_markers:
            if not m.id in marker_by_id or m.perimeter > marker_by_id[m.id].perimeter:
                marker_by_id[m.id] = m

        return marker_by_id

    def _findHomographies(self, points_A, points_B):
        points_B.shape = -1, 1, 2
        points_A.shape = -1, 1, 2

        B_to_A, mask = cv2.findHomography(
            points_A, points_B, method=cv2.RANSAC, ransacReprojThreshold=100
        )

        if not mask.all():
            return None, None

        A_to_B, mask = cv2.findHomography(points_B, points_A)
        return A_to_B, B_to_A

    def _generate_heatmap(self):
        if not self.gaze_history:
            return

        data = [x['gaze'] for x in self.gaze_history]

        grid = int(self.real_world_size["y"]), int(self.real_world_size["x"])

        xvals, yvals = zip(*((x, 1. - y) for x, y in data))
        hist, *edges = np.histogram2d(
            yvals, xvals, bins=grid, range=[[0, 1.], [0, 1.]], normed=False
        )
        filter_h = int(self.heatmap_detail * grid[0]) // 2 * 2 + 1
        filter_w = int(self.heatmap_detail * grid[1]) // 2 * 2 + 1
        hist = cv2.GaussianBlur(hist, (filter_h, filter_w), 0)

        hist_max = hist.max()
        hist *= (255. / hist_max) if hist_max else 0.
        hist = hist.astype(np.uint8)
        c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
        # reuse allocated memory if possible
        if self.heatmap.shape != (*grid, 4):
            self.heatmap = np.ones((*grid, 4), dtype=np.uint8)
            self.heatmap[:, :, 3] = 125
        self.heatmap[:, :, :3] = c_map

    def save_to_dict(self):
        return {
            'name': self.name,
            'real_world_size': self.real_world_size,
            'reg_markers': [marker.save_to_dict() for marker in
                            self.reg_markers.values()],
            'build_up_status': self.build_up_status
        }

    def load_from_dict(self, init_dict):
        self.name = init_dict['name']
        self.real_world_size = init_dict['real_world_size']
        self.reg_markers = [_Surface_Marker(marker['id'], verts=marker['verts']) for
                            marker in init_dict['reg_markers']]
        self.reg_markers = {m.id: m for m in self.reg_markers}
        self.build_up_status = init_dict['build_up_status']
        self.defined = True

    def cleanup(self):
        pass # TODO implement
        # if self._window:
        #     self.close_window()

class _Surface_Marker(object):
    """
    A Surface Marker is located in normalized surface space, unlike regular Markers which are
    located in image space. It's location on the surface is aggregated over a list of
    obersvations.
    """

    def __init__(self, id, verts=None):
        self.id = id
        self.verts = None
        self.observations = []

        if not verts is None:
            self.verts = np.array(verts)

    def load_uv_coords(self, uv_coords): # TODO Where is this used?
        self.verts = uv_coords

    def add_observation(self, uv_coords):
        self.observations.append(uv_coords)
        self._compute_robust_mean()

    def _compute_robust_mean(self):
        # uv is of shape (N, 4, 2) where N is the number of collected observations
        uv = np.array(self.observations)
        base_line_mean = np.mean(uv, axis=0)
        distance = np.linalg.norm(uv - base_line_mean, axis=(1,2))

        # Estimate the mean again using the 50% closest samples
        cut_off = sorted(distance)[len(distance) // 2]
        uv_subset = uv[distance <= cut_off]
        final_mean = np.mean(uv_subset, axis=0)
        self.verts = final_mean

    def save_to_dict(self):
        return {
            'id': self.id,
            'verts': [v.tolist() for v in self.verts]
        }