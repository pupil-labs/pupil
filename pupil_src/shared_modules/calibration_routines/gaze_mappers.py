"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from collections import deque

import cv2
import numpy as np
from pyglui import ui

import math_helper
from plugin import Plugin
from methods import normalize

from . import calibrate
from .visualizer_calibration import Calibration_Visualizer


def _clamp_norm_point(pos):
    """realistic numbers for norm pos should be in this range.
        Grossly bigger or smaller numbers are results bad exrapolation
        and can cause overflow erorr when denormalized and cast as int32.
    """
    return min(100.0, max(-100.0, pos[0])), min(100.0, max(-100.0, pos[1]))


class Gaze_Mapping_Plugin(Plugin):
    """base class for all gaze mapping routines"""

    uniqueness = "by_base_class"
    order = 0.1
    icon_chr = chr(0xEC20)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.active_gaze_mapping_plugin = self

    def on_pupil_datum(self, p):
        raise NotImplementedError()

    def map_batch(self, pupil_list):
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))
        return results

    def add_menu(self):
        super().add_menu()
        self.menu_icon.order = 0.31


class Monocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def on_pupil_datum(self, p):
        g = self._map_monocular(p)
        if g:
            return [g]
        else:
            return []


class Binocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""

    def __init__(
        self,
        g_pool,
        min_pupil_confidence=0.6,
        temporal_cutoff=0.03,
        timestamp_reset_threshold=5.0,
        process_low_confidence=True,
        sample_cutoff=10,
    ):
        super().__init__(g_pool)
        self.min_pupil_confidence = min_pupil_confidence
        self.temporal_cutoff = temporal_cutoff
        self.timestamp_reset_threshold = timestamp_reset_threshold
        self.process_low_confidence = process_low_confidence
        self.sample_cutoff = sample_cutoff
        self._cache = deque()

    def _reset(self):
        self._cache.clear()

    def add_menu(self):
        super().add_menu()
        self.menu.append(
            ui.Slider(
                "min_pupil_confidence",
                self,
                step=0.01,
                min=0.0,
                max=1.0,
                label="Min conf. for binocular",
            )
        )
        self.menu.append(
            ui.Slider(
                "temporal_cutoff",
                self,
                step=0.002,
                min=0.0,
                max=0.2,
                label="Max binocular time diff. (s)",
            )
        )
        self.menu.append(
            ui.Slider(
                "sample_cutoff", self, step=1, min=2, max=20, label="Max queue length"
            )
        )

    def map_batch(self, pupil_list):
        backup_cache = self._cache
        self._cache = deque()
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))
        self._cache = backup_cache
        return results

    def on_pupil_datum(self, p):
        """
        Process pupil datum, combine with data in cache if possible, and return gaze datum.
        :param p: pupil datum (message with topic 'pupil.0' or 'pupil.1')
        :return: List of gaze_data or [].

        Note the mapping logic has recently changed. See https://github.com/pupil-labs/pupil/pull/1557

        One objective of the new logic is to never push a sample with an older timestamp than a sample that has already
        been pushed. This is not simple to achieve because sometimes a new sample will arrive (e.g. from eye B) with an
        older timestamp than a previous sample (e.g. from eye A). i.e. A0-A1-A2-B0-A3-B1-A4-B2-A5 etc.
        If we wish to combine samples from the two eyes for binocular reconstruction then we cannot push samples
        immediately upon receipt and we must maintain a buffer of samples. How to do that without incurring too much
        delay is tricky.

        The only thing we can be sure of is that a new sample from one eye will have a later timestamp than
        previous samples from that same eye.
        """
        # Reset state if we receive a new timestamp that is too old (probably due to clock reset).
        if len(self._cache) > 0:
            t_delta = np.abs(p["timestamp"] - self._cache[-1]["timestamp"])
            if t_delta > self.timestamp_reset_threshold:
                self._reset()

        # Insert the new sample in the cache according to its timestamp.
        if (len(self._cache) == 0) or (p["timestamp"] > self._cache[-1]["timestamp"]):
            # Shortcut when timestamp is newer than anything in cache.
            self._cache.append(p)
        else:
            cache_ts = [_["timestamp"] for _ in self._cache]
            in_ix = np.searchsorted(cache_ts, p["timestamp"])
            self._cache.insert(in_ix, p)

        # Process the cache left to right, stopping when we cannot confidently pop a sample.
        output = []
        for ix in range(len(self._cache) - 1):
            # We can only ever pop a sample when there is a newer sample from the opposite eye,
            # because then we know there cannot be a sample inserted older than the current sample.
            newer_other = -1
            do_binoc = False

            for ix, other_samp in enumerate(self._cache):
                if other_samp["id"] == 1 - self._cache[0]["id"]:
                    newer_other = ix

                    # If current sample is low-conf then we don't need to search anymore. _Any_ other-eye samp is OK.
                    if not self._cache[0]["confidence"]:
                        break

                    # If newer_other is more than temporal_cutoff newer then we don't need to search anymore.
                    time_diff = other_samp["timestamp"] - self._cache[0]["timestamp"]
                    if time_diff > self.temporal_cutoff:
                        break

                    # We know from above that current sample is high-conf and newer_other is within temporal_cutoff.
                    # If newer_other is also high-confidence then we can do binocular mapping.
                    if other_samp["confidence"] >= self.min_pupil_confidence:
                        do_binoc = True
                        break

            # If we have failed to find a newer sample from the other eye
            # then we cannot do anything because the other eye might yet insert a sample older than the current sample.
            # The only exception is if the cache is overflowing, then do _map_monocular anyway.
            if (newer_other < 0) and (len(self._cache) <= self.sample_cutoff):
                break

            if do_binoc:
                if self._cache[0]["id"] == 0:
                    mapped_sample = self._map_binocular(
                        self._cache[0], self._cache[newer_other]
                    )
                else:
                    mapped_sample = self._map_binocular(
                        self._cache[newer_other], self._cache[0]
                    )
            else:
                mapped_sample = self._map_monocular(self._cache[0])
            output.append(mapped_sample)

            # Pop the first sample off our cache
            self._cache.popleft()

            # Cannot continue unless we have more than 1 sample in the cache.
            if len(self._cache) == 1:
                break

        return output


class Dummy_Gaze_Mapper(Monocular_Gaze_Mapper_Base, Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def _map_monocular(self, p):
        return {
            "topic": "gaze.2d.{}.".format(p["id"]),
            "norm_pos": p["norm_pos"],
            "confidence": p["confidence"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Dummy gaze mapper"
        self.menu.append(ui.Info_Text("Please calibrate."))

    def deinit_ui(self):
        self.remove_menu()


class Monocular_Gaze_Mapper(Monocular_Gaze_Mapper_Base, Gaze_Mapping_Plugin):
    """docstring for Monocular_Gaze_Mapper"""

    def __init__(self, g_pool, params):
        super().__init__(g_pool)
        self.params = params
        self.map_fn = calibrate.make_map_function(*self.params)

    def _map_monocular(self, p):
        gaze_point = self.map_fn(p["norm_pos"])
        return {
            "topic": "gaze.2d.{}.".format(p["id"]),
            "norm_pos": gaze_point,
            "confidence": p["confidence"],
            "id": p["id"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

    def get_init_dict(self):
        return {"params": self.params}


class Dual_Monocular_Gaze_Mapper(Monocular_Gaze_Mapper_Base, Gaze_Mapping_Plugin):
    """A gaze mapper that maps two eyes individually"""

    def __init__(self, g_pool, params0, params1):
        super().__init__(g_pool)
        self.params0 = params0
        self.params1 = params1
        self.map_fns = (
            calibrate.make_map_function(*self.params0),
            calibrate.make_map_function(*self.params1),
        )

    def _map_monocular(self, p):
        gaze_point = self.map_fns[p["id"]](p["norm_pos"])
        return {
            "topic": "gaze.2d.{}.".format(p["id"]),
            "norm_pos": gaze_point,
            "confidence": p["confidence"],
            "id": p["id"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

    def get_init_dict(self):
        return {"params0": self.params0, "params1": self.params1}


class Binocular_Gaze_Mapper(Binocular_Gaze_Mapper_Base, Gaze_Mapping_Plugin):
    def __init__(self, g_pool, params, params_eye0, params_eye1):
        super().__init__(g_pool)
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = True
        self.map_fn = calibrate.make_map_function(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(calibrate.make_map_function(*self.params_eye0))
        self.map_fn_fallback.append(calibrate.make_map_function(*self.params_eye1))

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Binocular Gaze Mapper"
        self.menu.append(ui.Switch("multivariate", self, label="Multivariate Mode"))

    def deinit_ui(self):
        self.remove_menu()

    def _map_binocular(self, p0, p1):
        if self.multivariate:
            gaze_point = self.map_fn(p0["norm_pos"], p1["norm_pos"])
        else:
            gaze_point_eye0 = self.map_fn_fallback[0](p0["norm_pos"])
            gaze_point_eye1 = self.map_fn_fallback[1](p1["norm_pos"])
            gaze_point = (
                (gaze_point_eye0[0] + gaze_point_eye1[0]) / 2.0,
                (gaze_point_eye0[1] + gaze_point_eye1[1]) / 2.0,
            )
        confidence = (p0["confidence"] + p1["confidence"]) / 2.0
        ts = min(
            p0["timestamp"], p1["timestamp"]
        )  # Was prev. avg, but min is more consistent with mapper logic.
        return {
            "topic": "gaze.2d.01.",
            "norm_pos": gaze_point,
            "confidence": confidence,
            "timestamp": ts,
            "base_data": [p0, p1],
        }

    def _map_monocular(self, p):
        gaze_point = self.map_fn_fallback[p["id"]](p["norm_pos"])
        return {
            "topic": "gaze.2d.{}.".format(p["id"]),
            "norm_pos": gaze_point,
            "confidence": p["confidence"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

    def get_init_dict(self):
        return {
            "params": self.params,
            "params_eye0": self.params_eye0,
            "params_eye1": self.params_eye1,
        }


class Vector_Gaze_Mapper(Monocular_Gaze_Mapper_Base, Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""

    def __init__(
        self,
        g_pool,
        eye_camera_to_world_matrix,
        cal_points_3d,
        cal_ref_points_3d,
        cal_gaze_points_3d,
        gaze_distance=500,
    ):
        super().__init__(g_pool)
        self.eye_camera_to_world_matrix = np.asarray(eye_camera_to_world_matrix)
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3, :3]
        self.rotation_vector = cv2.Rodrigues(self.rotation_matrix)[0]
        self.translation_vector = self.eye_camera_to_world_matrix[:3, 3]
        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d
        self.cal_gaze_points_3d = cal_gaze_points_3d
        self.g_pool = g_pool
        self.gaze_pts_debug = []
        self.sphere = {}
        self.gaze_distance = gaze_distance

    def toWorld(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix, point)[:3]

    def init_ui(self):
        self.add_menu()
        self.visualizer = Calibration_Visualizer(
            self.g_pool,
            self.cal_points_3d,
            self.cal_ref_points_3d,
            self.eye_camera_to_world_matrix,
            self.cal_gaze_points_3d,
        )

        def open_close_window(new_state):
            if new_state:
                self.visualizer.open_window()
            else:
                self.visualizer.close_window()

        self.menu.label = "Monocular 3D gaze mapper"
        self.menu.append(
            ui.Switch(
                "debug window",
                setter=open_close_window,
                getter=lambda: bool(self.visualizer.window),
            )
        )
        self.menu.append(
            ui.Slider("gaze_distance", self, min=50, max=2000, label="gaze distance mm")
        )

    def _map_monocular(self, p):
        if "3d" not in p["method"]:
            return None

        gaze_point = np.array(p["circle_3d"]["normal"]) * self.gaze_distance + np.array(
            p["sphere"]["center"]
        )

        image_point = self.g_pool.capture.intrinsics.projectPoints(
            np.array([gaze_point]), self.rotation_vector, self.translation_vector
        )
        image_point = image_point.reshape(-1, 2)
        image_point = normalize(
            image_point[0], self.g_pool.capture.intrinsics.resolution, flip_y=True
        )
        image_point = _clamp_norm_point(image_point)

        eye_center = self.toWorld(p["sphere"]["center"])
        gaze_3d = self.toWorld(gaze_point)
        normal_3d = np.dot(self.rotation_matrix, np.array(p["circle_3d"]["normal"]))

        g = {
            "topic": "gaze.3d.{}.".format(p["id"]),
            "norm_pos": image_point,
            "eye_center_3d": eye_center.tolist(),
            "gaze_normal_3d": normal_3d.tolist(),
            "gaze_point_3d": gaze_3d.tolist(),
            "confidence": p["confidence"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

        if hasattr(self, "visualizer") and self.visualizer.window:
            self.gaze_pts_debug.append(gaze_3d)
            self.sphere["center"] = eye_center  # eye camera coordinates
            self.sphere["radius"] = p["sphere"]["radius"]
        return g

    def gl_display(self):
        self.visualizer.update_window(self.g_pool, self.gaze_pts_debug, self.sphere)
        self.gaze_pts_debug = []

    def get_init_dict(self):
        return {
            "eye_camera_to_world_matrix": self.eye_camera_to_world_matrix.tolist(),
            "cal_points_3d": self.cal_points_3d,
            "cal_ref_points_3d": self.cal_ref_points_3d,
            "cal_gaze_points_3d": self.cal_gaze_points_3d,
            "gaze_distance": self.gaze_distance,
        }

    def deinit_ui(self):
        self.remove_menu()
        self.visualizer.close_window()


class Binocular_Vector_Gaze_Mapper(Binocular_Gaze_Mapper_Base, Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""

    def __init__(
        self,
        g_pool,
        eye_camera_to_world_matrix0,
        eye_camera_to_world_matrix1,
        cal_points_3d=[],
        cal_ref_points_3d=[],
        cal_gaze_points0_3d=[],
        cal_gaze_points1_3d=[],
        backproject=True,
    ):
        super().__init__(g_pool)

        self.backproject = backproject
        self.eye_camera_to_world_matricies = (
            np.asarray(eye_camera_to_world_matrix0),
            np.asarray(eye_camera_to_world_matrix1),
        )
        self.rotation_matricies = (
            self.eye_camera_to_world_matricies[0][:3, :3],
            self.eye_camera_to_world_matricies[1][:3, :3],
        )
        self.rotation_vectors = (
            cv2.Rodrigues(self.eye_camera_to_world_matricies[0][:3, :3])[0],
            cv2.Rodrigues(self.eye_camera_to_world_matricies[1][:3, :3])[0],
        )
        self.translation_vectors = (
            self.eye_camera_to_world_matricies[0][:3, 3],
            self.eye_camera_to_world_matricies[1][:3, 3],
        )

        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d

        self.cal_gaze_points0_3d = cal_gaze_points0_3d  # save for debug window
        self.cal_gaze_points1_3d = cal_gaze_points1_3d  # save for debug window

        self.g_pool = g_pool
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []
        self.sphere0 = {}
        self.sphere1 = {}
        self.last_gaze_distance = 500.0

    def eye0_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[0], point)[:3]

    def eye1_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[1], point)[:3]

    def init_ui(self):
        self.add_menu()
        self.visualizer = Calibration_Visualizer(
            self.g_pool,
            cal_ref_points_3d=self.cal_points_3d,
            cal_observed_points_3d=self.cal_ref_points_3d,
            eye_camera_to_world_matrix0=self.eye_camera_to_world_matricies[0],
            cal_gaze_points0_3d=self.cal_gaze_points0_3d,
            eye_camera_to_world_matrix1=self.eye_camera_to_world_matricies[1],
            cal_gaze_points1_3d=self.cal_gaze_points1_3d,
        )

        def open_close_window(new_state):
            if new_state:
                self.visualizer.open_window()
            else:
                self.visualizer.close_window()

        self.menu.label = "Binocular 3D gaze mapper"
        # self.menu.append(ui.Text_Input('last_gaze_distance',self))
        self.menu.append(
            ui.Switch(
                "debug window",
                setter=open_close_window,
                getter=lambda: bool(self.visualizer.window),
            )
        )

    def deinit_ui(self):
        self.remove_menu()
        self.visualizer.close_window()

    def _map_monocular(self, p):
        if "3d" not in p["method"]:
            return None

        p_id = p["id"]
        gaze_point = np.array(
            p["circle_3d"]["normal"]
        ) * self.last_gaze_distance + np.array(p["sphere"]["center"])

        if self.backproject:
            image_point = self.g_pool.capture.intrinsics.projectPoints(
                np.array([gaze_point]),
                self.rotation_vectors[p_id],
                self.translation_vectors[p_id],
            )
            image_point = image_point.reshape(-1, 2)
            image_point = normalize(
                image_point[0], self.g_pool.capture.intrinsics.resolution, flip_y=True
            )
            image_point = _clamp_norm_point(image_point)

        if p_id == 0:
            eye_center = self.eye0_to_World(p["sphere"]["center"])
            gaze_3d = self.eye0_to_World(gaze_point)
        else:
            eye_center = self.eye1_to_World(p["sphere"]["center"])
            gaze_3d = self.eye1_to_World(gaze_point)

        normal_3d = self.rotation_matricies[p_id] @ np.array(p["circle_3d"]["normal"])

        g = {
            "topic": "gaze.3d.{}.".format(p_id),
            "eye_center_3d": eye_center.tolist(),
            "gaze_normal_3d": normal_3d.tolist(),
            "gaze_point_3d": gaze_3d.tolist(),
            "confidence": p["confidence"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

        if self.backproject:
            g["norm_pos"] = image_point

        if hasattr(self, "visualizer") and self.visualizer.window:
            if p_id == 0:
                self.gaze_pts_debug0.append(gaze_3d)
                self.sphere0["center"] = eye_center
                self.sphere0["radius"] = p["sphere"]["radius"]
            else:
                self.gaze_pts_debug1.append(gaze_3d)
                self.sphere1["center"] = eye_center
                self.sphere1["radius"] = p["sphere"]["radius"]

        return g

    def _map_binocular(self, p0, p1):

        if "3d" not in p0["method"] or "3d" not in p1["method"]:
            return None

        # find the nearest intersection point of the two gaze lines
        # eye ball centers in world coords
        s0_center = self.eye0_to_World(np.array(p0["sphere"]["center"]))
        s1_center = self.eye1_to_World(np.array(p1["sphere"]["center"]))
        # eye line of sight in world coords
        s0_normal = np.dot(
            self.rotation_matricies[0], np.array(p0["circle_3d"]["normal"])
        )
        s1_normal = np.dot(
            self.rotation_matricies[1], np.array(p1["circle_3d"]["normal"])
        )

        # See Lech Swirski: "Gaze estimation on glasses-based stereoscopic displays"
        # Chapter: 7.4.2 Cyclopean gaze estimate

        # the cyclop is the avg of both lines of sight
        cyclop_normal = (s0_normal + s1_normal) / 2.0
        cyclop_center = (s0_center + s1_center) / 2.0

        # We use it to define a viewing plane.
        gaze_plane = np.cross(cyclop_normal, s1_center - s0_center)
        gaze_plane = gaze_plane / np.linalg.norm(gaze_plane)

        # project lines of sight onto the gaze plane
        s0_norm_on_plane = s0_normal - np.dot(gaze_plane, s0_normal) * gaze_plane
        s1_norm_on_plane = s1_normal - np.dot(gaze_plane, s1_normal) * gaze_plane

        # create gaze lines on this plane
        gaze_line0 = [s0_center, s0_center + s0_norm_on_plane]
        gaze_line1 = [s1_center, s1_center + s1_norm_on_plane]

        # find the intersection of left and right line of sight.
        nearest_intersection_point, intersection_distance = math_helper.nearest_intersection(
            gaze_line0, gaze_line1
        )
        if nearest_intersection_point is not None and self.backproject:
            cyclop_gaze = nearest_intersection_point - cyclop_center
            self.last_gaze_distance = np.sqrt(cyclop_gaze.dot(cyclop_gaze))
            image_point = self.g_pool.capture.intrinsics.projectPoints(
                np.array([nearest_intersection_point])
            )
            image_point = image_point.reshape(-1, 2)
            image_point = normalize(
                image_point[0], self.g_pool.capture.intrinsics.resolution, flip_y=True
            )
            image_point = _clamp_norm_point(image_point)

        if hasattr(self, "visualizer") and self.visualizer.window:
            gaze0_3d = s0_normal * self.last_gaze_distance + s0_center
            gaze1_3d = s1_normal * self.last_gaze_distance + s1_center
            self.gaze_pts_debug0.append(gaze0_3d)
            self.gaze_pts_debug1.append(gaze1_3d)
            if nearest_intersection_point is not None:
                self.intersection_points_debug.append(nearest_intersection_point)

            self.sphere0["center"] = s0_center  # eye camera coordinates
            self.sphere0["radius"] = p0["sphere"]["radius"]
            self.sphere1["center"] = s1_center  # eye camera coordinates
            self.sphere1["radius"] = p1["sphere"]["radius"]

        if nearest_intersection_point is None:
            return None

        confidence = min(p0["confidence"], p1["confidence"])
        ts = min(p0["timestamp"], p1["timestamp"])
        g = {
            "topic": "gaze.3d.01.",
            "eye_centers_3d": {0: s0_center.tolist(), 1: s1_center.tolist()},
            "gaze_normals_3d": {0: s0_normal.tolist(), 1: s1_normal.tolist()},
            "gaze_point_3d": nearest_intersection_point.tolist(),
            "confidence": confidence,
            "timestamp": ts,
            "base_data": [p0, p1],
        }

        if self.backproject:
            g["norm_pos"] = image_point

        return g

    def gl_display(self):
        self.visualizer.update_window(
            self.g_pool,
            self.gaze_pts_debug0,
            self.sphere0,
            self.gaze_pts_debug1,
            self.sphere1,
            self.intersection_points_debug,
        )
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []

    def get_init_dict(self):
        return {
            "eye_camera_to_world_matrix0": self.eye_camera_to_world_matricies[
                0
            ].tolist(),
            "eye_camera_to_world_matrix1": self.eye_camera_to_world_matricies[
                1
            ].tolist(),
            "cal_ref_points_3d": self.cal_ref_points_3d,
            "cal_gaze_points0_3d": self.cal_gaze_points0_3d,
            "cal_gaze_points1_3d": self.cal_gaze_points1_3d,
            "backproject": self.backproject,
        }

    def cleanup(self):
        super().cleanup()
        if hasattr(self, "visualizer"):
            self.visualizer.close_window()
