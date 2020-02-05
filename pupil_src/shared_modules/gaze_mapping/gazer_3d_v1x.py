"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import typing as T

import cv2
import numpy as np

from gaze_mapping.gazer_base import (
    GazerBase,
    Model,
    data_processing,
    NotEnoughDataError,
    FitDidNotConvergeError,
)

from calibration_routines.gaze_mappers import _clamp_norm_point, normalize
from calibration_routines.optimization_calibration import utils
from calibration_routines.optimization_calibration.calibrate_3d import (
    calibrate_binocular,
    calibrate_monocular,
)


class Model3D_v1x(Model):
    @abc.abstractmethod
    def _fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _predict_single(self, x):
        pass

    def __init__(self, *, intrinsics):
        self.intrinsics = intrinsics
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, extracted_data):  # NOTE: breaks api due to legacy reasons
        params = self._fit(*extracted_data)
        self.set_params(**params)
        self._is_fitted = True

    def predict(self, X):
        predictions = (self._predict_single(x) for x in X)
        predictions = filter(bool, predictions)
        return predictions

    def set_params(self, **params):
        self._params = params

    def get_params(self):
        return {"params": self._params}


class Model3D_v1x_Monocular(Model3D_v1x):
    def _fit(
        self, unprojected_ref_points, pupil_normals, last_pupil, initial_depth=500
    ):
        pupil_id = last_pupil["id"]
        result = calibrate_monocular(
            unprojected_ref_points, pupil_normals, pupil_id, initial_depth
        )
        success, poses_in_world, gaze_targets_in_world = result
        if not success:
            raise FitDidNotConvergeError

        world_pose, eye_pose = poses_in_world

        sphere_pos = last_pupil["sphere"]["center"]
        eye_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye_pose, sphere_pos)

        all_observations = [unprojected_ref_points, pupil_normals]
        nearest_points = utils.calculate_nearest_points_to_targets(
            all_observations, poses_in_world, gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye = nearest_points

        params = {
            "eye_camera_to_world_matrix": eye_cam_pose_in_world.tolist(),
            "cal_points_3d": gaze_targets_in_world.tolist(),
            "cal_ref_points_3d": nearest_points_world.tolist(),
            "cal_gaze_points_3d": nearest_points_eye.tolist(),
            "gaze_distance": initial_depth,
        }
        return params

    def set_params(self, **params):
        super().set_params(**params)
        self.cal_points_3d = params["cal_points_3d"]
        self.cal_ref_points_3d = params["cal_ref_points_3d"]
        self.cal_gaze_points_3d = params["cal_gaze_points_3d"]
        self.gaze_distance = params["gaze_distance"]

        self.eye_camera_to_world_matrix = np.asarray(
            params["eye_camera_to_world_matrix"]
        )
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3, :3]
        self.rotation_vector = cv2.Rodrigues(self.rotation_matrix)[0]
        self.translation_vector = self.eye_camera_to_world_matrix[:3, 3]

    def _predict_single(self, p):
        gaze_point = np.array(p["circle_3d"]["normal"]) * self.gaze_distance + np.array(
            p["sphere"]["center"]
        )

        image_point = self.intrinsics.projectPoints(
            np.array([gaze_point]), self.rotation_vector, self.translation_vector
        )
        image_point = image_point.reshape(-1, 2)
        image_point = normalize(image_point[0], self.intrinsics.resolution, flip_y=True)
        image_point = _clamp_norm_point(image_point)

        eye_center = self._toWorld(p["sphere"]["center"])
        gaze_3d = self._toWorld(gaze_point)
        normal_3d = np.dot(self.rotation_matrix, np.array(p["circle_3d"]["normal"]))

        g = {
            "norm_pos": image_point,
            "eye_center_3d": eye_center.tolist(),
            "gaze_normal_3d": normal_3d.tolist(),
            "gaze_point_3d": gaze_3d.tolist(),
        }
        return g

    def _toWorld(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix, point)[:3]


class Model3D_v1x_Binocular(Model3D_v1x):
    def _fit(
        self,
        unprojected_ref_points,
        pupil0_normals,
        pupil1_normals,
        last_pupil0,
        last_pupil1,
        initial_depth=500,
    ):

        res = calibrate_binocular(
            unprojected_ref_points, pupil0_normals, pupil1_normals, initial_depth
        )
        success, poses_in_world, gaze_targets_in_world = res
        if not success:
            raise FitDidNotConvergeError

        world_pose, eye0_pose, eye1_pose = poses_in_world

        sphere_pos0 = last_pupil0["sphere"]["center"]
        sphere_pos1 = last_pupil1["sphere"]["center"]
        eye0_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
        eye1_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

        all_observations = [unprojected_ref_points, pupil0_normals, pupil1_normals]
        nearest_points = utils.calculate_nearest_points_to_targets(
            all_observations, poses_in_world, gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

        params = {
            "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
            "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
            "cal_points_3d": gaze_targets_in_world.tolist(),
            "cal_ref_points_3d": nearest_points_world.tolist(),
            "cal_gaze_points0_3d": nearest_points_eye0.tolist(),
            "cal_gaze_points1_3d": nearest_points_eye1.tolist(),
        }
        return params

    def set_params(self, **params):
        super().set_params(**params)
        self.last_gaze_distance = 500.0

        # save for debug window
        self.cal_points_3d = params["cal_points_3d"]
        self.cal_ref_points_3d = params["cal_ref_points_3d"]
        self.cal_gaze_points0_3d = params["cal_gaze_points0_3d"]
        self.cal_gaze_points1_3d = params["cal_gaze_points1_3d"]

        self.backproject = params.get("backproject", True)
        self.eye_camera_to_world_matricies = (
            np.asarray(params["eye_camera_to_world_matrix0"]),
            np.asarray(params["eye_camera_to_world_matrix1"]),
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

    def _predict_single(self, p):
        # find the nearest intersection point of the two gaze lines
        # eye ball centers in world coords
        s0_center = self._eye0_to_World(np.array(p0["sphere"]["center"]))
        s1_center = self._eye1_to_World(np.array(p1["sphere"]["center"]))
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
        (
            nearest_intersection_point,
            intersection_distance,
        ) = math_helper.nearest_intersection(gaze_line0, gaze_line1)
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

        if nearest_intersection_point is None:
            return None

        g = {
            "eye_centers_3d": {0: s0_center.tolist(), 1: s1_center.tolist()},
            "gaze_normals_3d": {0: s0_normal.tolist(), 1: s1_normal.tolist()},
            "gaze_point_3d": nearest_intersection_point.tolist(),
        }

        if self.backproject:
            g["norm_pos"] = image_point

        return g

    def _eye0_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[0], point)[:3]

    def _eye1_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[1], point)[:3]


class Gazer3D_v1x(GazerBase):
    label = "3D (v1)"

    def _init_left_model(self) -> Model:
        return Model3D_v1x_Monocular(intrinsics=self.g_pool.capture.intrinsics)

    def _init_right_model(self) -> Model:
        return Model3D_v1x_Monocular(intrinsics=self.g_pool.capture.intrinsics)

    def _init_binocular_model(self) -> Model:
        return Model3D_v1x_Binocular(intrinsics=self.g_pool.capture.intrinsics)

    def _fit_monocular_model(self, model: Model, matched_data: T.Iterable):
        self._fit_model(model, matched_data)

    def _fit_binocular_model(self, model: Model, matched_data: T.Iterable):
        self._fit_model(model, matched_data)

    def _fit_model(self, model: Model, matched_data: T.Iterable):
        # 3d monocular/binocular calls the same function for data extraction
        extracted_data = data_processing._extract_3d_data(
            matched_data, self.g_pool.capture.intrinsics
        )
        if not extracted_data:
            raise NotEnoughDataError
        model.fit(extracted_data)

    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)

            if num_matched == 2 and self.binocular_model.is_fitted:
                gaze_positions = self.binocular_model.predict([pupil_match])
                topic = "gaze.3d.01."
            elif num_matched == 1:
                if pupil_match[0]["id"] == 0 and self.right_model.is_fitted:
                    gaze_positions = self.right_model.predict([pupil_match])
                    topic = "gaze.3d.0."
                elif pupil_match[0]["id"] == 1 and self.left_model.is_fitted:
                    gaze_positions = self.left_model.predict([pupil_match])
                    topic = "gaze.3d.1."

            for gaze_pos in gaze_positions:
                gaze_pos.update(
                    {
                        "topic": topic,
                        "confidence": np.mean([p["confidence"] for p in pupil_match]),
                        "timestamp": np.mean([p["timestamp"] for p in pupil_match]),
                        "base_data": pupil_match,
                    }
                )
                yield gaze_pos

    def filter_pupil_data(
        self, pupil_data: T.Iterable, confidence_threshold: T.Optional[float] = None
    ) -> T.Iterable:
        # TODO: Filter 3D data
        pupil_data = super().filter_pupil_data(pupil_data, confidence_threshold)
        return pupil_data
