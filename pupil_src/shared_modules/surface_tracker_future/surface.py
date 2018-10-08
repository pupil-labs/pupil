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
    def __init__(self, init_dict=None):
        """
        # TODO create docstring
        Args:
            marker_min_perimeter:
            marker_min_confidence:
            init_dict:
        """
        self.uid = random.randint(0, 1e6)

        self.name = "unknown"
        self.real_world_size = {"x": 1., "y": 1.}

        # TODO move this into class docstring
        # We store the surface state in two versions: once computed with the
        # undistorted scene image and once with the still distorted scene image. The
        # undistorted state is used to map gaze onto the surface, the distorted one
        # is used for visualization. This is necessary because surface corners
        # outside of the image can not be re-distorted for visualization correctly.
        # Instead the slightly wrong but correct looking distorted version is
        # used for visualization.
        self.reg_markers_undist = {}
        self.reg_markers_dist = {}
        self.img_to_surf_trans = None
        self.surf_to_img_trans = None
        self.dist_img_to_surf_trans = None
        self.surf_to_dist_img_trans = None

        self.detected = False
        self.num_detected_markers = []

        self._required_obs_per_marker = 5
        self._avg_obs_per_marker = 0
        self.build_up_status = 0

        self.gaze_history_length = 1
        self.gaze_history = collections.deque()
        self.heatmap = np.ones((1, 1), dtype=np.uint8)
        self.heatmap_detail = .2

        if not init_dict is None:
            self.load_from_dict(init_dict)

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        if isinstance(other, Surface):
            return self.uid == other.uid
        else:
            return False

    @property
    def defined(self):
        return self.build_up_status >= 1.0

    def map_to_surf(self, points, camera_model, compensate_distortion=True):
        """Map points from image space to normalized surface space.

        Args:
            points (ndarray): An array of points in one of the following shapes: (2,
            ) or (N, 2).
            camera_model: Camera Model object.
            compensate_distortion: If `True` camera distortion will be correctly
            compensated using the `camera_model`. Note that points that lie outside
            of the image can not be undistorted correctly and the attempt may
            introduce a large error.

        Returns:
            ndarray: Points mapped into normalized surface space. Has the same shape
            as the input.

        """
        orig_shape = points.shape

        if compensate_distortion:
            points = camera_model.undistortPoints(points)
        points.shape = (-1, 1, 2)

        if compensate_distortion:
            point_on_surf = cv2.perspectiveTransform(points, self.img_to_surf_trans)
        else:
            point_on_surf = cv2.perspectiveTransform(
                points, self.dist_img_to_surf_trans
            )
        point_on_surf.shape = orig_shape
        return point_on_surf

    def map_from_surf(self, points, camera_model, compensate_distortion=True):
        """Map points from normalized surface space to image space.

        Args:
            points (ndarray): An array of points in one of the following shapes: (2,
            ), (N, 2)
            camera_model: Camera Model object.
            compensate_distortion: If `True` camera distortion will be correctly
            compensated using the `camera_model`. Note that points that lie outside
            of the image can not be undistorted correctly and the attempt may
            introduce a large error.

        Returns:
            ndarray: Points mapped into image space. Has the same shape
            as the input.

        """
        orig_shape = points.shape
        points.shape = (-1, 1, 2)
        if compensate_distortion:
            img_points = cv2.perspectiveTransform(points, self.surf_to_img_trans)
        else:
            img_points = cv2.perspectiveTransform(points, self.surf_to_dist_img_trans)
        img_points.shape = orig_shape

        if compensate_distortion:
            img_points = camera_model.distortPoints(img_points)
        return img_points

    def move_corner(self, corner_idx, pos, camera_model):

        # Markers undistorted
        new_corner_pos = self.map_to_surf(pos, camera_model, compensate_distortion=True)
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[corner_idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for m in self.reg_markers_undist.values():
            m.verts = cv2.perspectiveTransform(
                m.verts.reshape((-1, 1, 2)), transform
            ).reshape((-1, 2))

        # Markers distorted
        new_corner_pos = self.map_to_surf(
            pos, camera_model, compensate_distortion=False
        )
        old_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)

        new_corners = old_corners.copy()
        new_corners[corner_idx] = new_corner_pos

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)
        for m in self.reg_markers_dist.values():
            m.verts = cv2.perspectiveTransform(
                m.verts.reshape((-1, 1, 2)), transform
            ).reshape((-1, 2))
        print(pos) # TODO delete

    def add_marker(self, id, verts, camera_model):
        surface_marker_dist = _Surface_Marker(id)
        marker_verts_dist = np.array(verts).reshape((4, 2))
        uv_coords_dist = self.map_to_surf(
            marker_verts_dist, camera_model, compensate_distortion=False
        )
        surface_marker_dist.add_observation(uv_coords_dist)
        self.reg_markers_dist[id] = surface_marker_dist

        surface_marker_undist = _Surface_Marker(id)
        marker_verts_undist = np.array(verts).reshape((4, 2))
        uv_coords_undist = self.map_to_surf(
            marker_verts_undist, camera_model, compensate_distortion=False
        )
        surface_marker_undist.add_observation(uv_coords_undist)
        self.reg_markers_undist[id] = surface_marker_undist

    def pop_marker(self, id):
        self.reg_markers_dist.pop(id)
        self.reg_markers_undist.pop(id)

    def update_location(self, idx, vis_markers, camera_model):
        vis_markers_dict = {m.id: m for m in vis_markers}

        if not self.defined:
            self.update_def(idx, vis_markers_dict, camera_model)

        # Get dict of current transformations
        transformations = self.locate(vis_markers_dict, camera_model, self.reg_markers_undist, self.reg_markers_dist)
        self.__dict__.update(transformations)

    def update_def(self, idx, vis_markers, camera_model):
        if not vis_markers:
            return

        all_verts_dist = np.array(
            [m.verts for m in vis_markers.values()], dtype=np.float32
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
            vis_markers.values(), marker_surf_coords_undist, marker_surf_coords_dist
        ):
            try:
                self.reg_markers_undist[m.id].add_observation(uv_undist)
                self.reg_markers_dist[m.id].add_observation(uv_dist)
            except KeyError:
                self.reg_markers_undist[m.id] = _Surface_Marker(m.id)
                self.reg_markers_undist[m.id].add_observation(uv_undist)
                self.reg_markers_dist[m.id] = _Surface_Marker(m.id)
                self.reg_markers_dist[m.id].add_observation(uv_dist)

        num_observations = sum(
            [len(m.observations) for m in self.reg_markers_undist.values()]
        )
        self._avg_obs_per_marker = num_observations / len(self.reg_markers_undist)
        self.build_up_status = self._avg_obs_per_marker / self._required_obs_per_marker

        if self.build_up_status >= 1:
            self._finalize_def()

    def _finalize_def(self):
        """
        # TODO improve docstring
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        persistent_markers_dist = {}
        for (k, m), m_dist in zip(
            self.reg_markers_undist.items(), self.reg_markers_dist.values()
        ):
            if len(m.observations) > self._required_obs_per_marker * .5:
                persistent_markers[k] = m
                persistent_markers_dist[k] = m_dist
        self.reg_markers_undist = persistent_markers
        self.reg_markers_dist = persistent_markers_dist

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

    @staticmethod
    def locate(vis_markers, camera_model, reg_markers_undist, reg_markers_dist):
        result = {
            'detected': False,
            'dist_img_to_surf_trans': None,
            'surf_to_dist_img_trans': None,
            'img_to_surf_trans': None,
            'surf_to_img_trans': None,
            'num_detected_markers': 0,
        }

        vis_reg_marker_ids = set(vis_markers.keys()) & set(
            reg_markers_undist.keys()
        )


        if not vis_reg_marker_ids or len(vis_reg_marker_ids) < min(
            2, len(reg_markers_undist)
        ):
            return result

        vis_verts_dist = np.array([vis_markers[id].verts for id in vis_reg_marker_ids])
        reg_verts_undist = np.array(
            [reg_markers_undist[id].verts for id in vis_reg_marker_ids]
        )
        reg_verts_dist = np.array(
            [reg_markers_dist[id].verts for id in vis_reg_marker_ids]
        )

        vis_verts_dist.shape = (-1, 2)
        reg_verts_undist.shape = (-1, 2)
        reg_verts_dist.shape = (-1, 2)

        dist_img_to_surf_trans, surf_to_dist_img_trans = Surface._findHomographies(
            reg_verts_dist, vis_verts_dist
        )

        vis_verts_undist = camera_model.undistortPoints(vis_verts_dist)
        img_to_surf_trans, surf_to_img_trans = Surface._findHomographies(
            reg_verts_undist, vis_verts_undist
        )

        if img_to_surf_trans is None or dist_img_to_surf_trans is None:
            return result
        else:
            result['detected'] = True
            result['dist_img_to_surf_trans'] = dist_img_to_surf_trans
            result['surf_to_dist_img_trans'] = surf_to_dist_img_trans
            result['img_to_surf_trans'] = img_to_surf_trans
            result['surf_to_img_trans'] = surf_to_img_trans
            result['num_detected_markers'] =  len(vis_reg_marker_ids),
            return result

    @staticmethod
    def _findHomographies(points_A, points_B):
        points_A = points_A.reshape((-1, 1, 2))
        points_B = points_B.reshape((-1, 1, 2))

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

        data = [x["gaze"] for x in self.gaze_history]

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

    def _denormalize(self, points, camera_model):
        if len(points.shape) == 1:
            points[1] = 1 - points[1]
        else:
            points[:, 1] = 1 - points[:, 1]
        points *= camera_model.resolution
        return points

    def on_change(self):
        pass

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

    def load_from_dict(self, init_dict):
        self.name = init_dict["name"]
        self.real_world_size = init_dict["real_world_size"]
        self.reg_markers_undist = [
            _Surface_Marker(marker["id"], verts=marker["verts"])
            for marker in init_dict["reg_markers"]
        ]
        self.reg_markers_undist = {m.id: m for m in self.reg_markers_undist}
        self.reg_markers_dist = [
            _Surface_Marker(marker["id"], verts=marker["verts"])
            for marker in init_dict["reg_markers_dist"]
        ]
        self.reg_markers_dist = {m.id: m for m in self.reg_markers_dist}
        self.build_up_status = init_dict["build_up_status"]


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

    def add_observation(self, uv_coords):
        self.observations.append(uv_coords)
        self._compute_robust_mean()

    def _compute_robust_mean(self):
        # uv is of shape (N, 4, 2) where N is the number of collected observations
        uv = np.array(self.observations)
        base_line_mean = np.mean(uv, axis=0)
        distance = np.linalg.norm(uv - base_line_mean, axis=(1, 2))

        # Estimate the mean again using the 50% closest samples
        cut_off = sorted(distance)[len(distance) // 2]
        uv_subset = uv[distance <= cut_off]
        final_mean = np.mean(uv_subset, axis=0)
        self.verts = final_mean

    def save_to_dict(self):
        return {"id": self.id, "verts": [v.tolist() for v in self.verts]}
