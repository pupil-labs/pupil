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
import logging
import typing as T

import cv2
import math_helper
import numpy as np
from gaze_mapping.gazer_base import (
    FitDidNotConvergeError,
    GazerBase,
    Model,
    NotEnoughDataError,
)
from methods import normalize

from .calibrate_3d import calibrate_binocular, calibrate_monocular
from .utils import (
    _clamp_norm_point,
    calculate_nearest_points_to_targets,
    get_eye_cam_pose_in_world,
)

logger = logging.getLogger(__name__)


_REFERENCE_FEATURE_COUNT = 3

_MONOCULAR_FEATURE_COUNT = 7
_MONOCULAR_EYEID = 0
_MONOCULAR_SPHERE_CENTER = slice(1, 4)
_MONOCULAR_PUPIL_NORMAL = slice(4, 7)

_BINOCULAR_FEATURE_COUNT = 14
_BINOCULAR_EYEID = 7
_BINOCULAR_SPHERE_CENTER = slice(8, 11)
_BINOCULAR_PUPIL_NORMAL = slice(11, 14)


class Model3D(Model):
    @abc.abstractmethod
    def _fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _predict_single(self, x):
        pass

    def __init__(self, *, intrinsics: T.Optional[T.Any], initial_depth: float):
        self.intrinsics = intrinsics
        self.initial_depth = initial_depth
        self._is_fitted = False
        self._params = {}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X, Y):
        assert X.ndim == Y.ndim == 2, (X.ndim, Y.ndim)
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[1] == _REFERENCE_FEATURE_COUNT
        params = self._fit(X, Y)
        self.set_params(**params)
        self._is_fitted = True

    def predict(self, X):
        assert X.ndim == 2
        predictions = (self._predict_single(x) for x in X)
        predictions = filter(bool, predictions)
        return predictions

    def set_params(self, **params):
        self._params = params
        self._is_fitted = True

    def get_params(self):
        return self._params


class Model3D_Monocular(Model3D):
    # optional access to binocular_model.last_gaze_distance
    binocular_model: "Model3D_Binocular" = None

    def __init__(self, *, initial_eye_translation, **kwargs):
        super().__init__(**kwargs)
        self._initial_eye_translation = np.array(initial_eye_translation)

    @property
    def gaze_distance(self):
        if self.binocular_model is not None and self.binocular_model.is_fitted:
            return self.binocular_model.last_gaze_distance
        return self._gaze_distance

    def _fit(self, X, Y):
        assert X.shape[1] == _MONOCULAR_FEATURE_COUNT
        unprojected_ref_points = Y
        pupil_normals = X[:, _MONOCULAR_PUPIL_NORMAL]

        pupil_id = X[-1, _MONOCULAR_EYEID]  # last pupil eye id
        sphere_pos = X[-1, _MONOCULAR_SPHERE_CENTER]  # last pupil sphere center

        result = calibrate_monocular(
            unprojected_ref_points,
            pupil_normals,
            pupil_id,
            self.initial_depth,
            initial_translation=self._initial_eye_translation,
        )
        success, poses_in_world, gaze_targets_in_world = result
        if not success:
            raise FitDidNotConvergeError

        world_pose, eye_pose = poses_in_world

        eye_cam_pose_in_world = get_eye_cam_pose_in_world(eye_pose, sphere_pos)

        all_observations = [unprojected_ref_points, pupil_normals]
        nearest_points = calculate_nearest_points_to_targets(
            all_observations, poses_in_world, gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye = nearest_points

        params = {
            "eye_camera_to_world_matrix": eye_cam_pose_in_world.tolist(),
            "gaze_distance": self.initial_depth,
        }
        return params

    def set_params(self, **params):
        super().set_params(**params)
        self._gaze_distance = params["gaze_distance"]

        self.eye_camera_to_world_matrix = np.asarray(
            params["eye_camera_to_world_matrix"]
        )
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3, :3]
        self.rotation_vector = cv2.Rodrigues(self.rotation_matrix)[0]
        self.translation_vector = self.eye_camera_to_world_matrix[:3, 3]

    def _predict_single(self, x):
        assert x.ndim == 1, x
        assert x.shape[0] == _MONOCULAR_FEATURE_COUNT, x
        pupil_normal = x[_MONOCULAR_PUPIL_NORMAL]
        sphere_center = x[_MONOCULAR_SPHERE_CENTER]
        gaze_point = pupil_normal * self.gaze_distance + sphere_center

        eye_center = self._toWorld(sphere_center)
        gaze_3d = self._toWorld(gaze_point)
        normal_3d = np.dot(self.rotation_matrix, pupil_normal)

        # Check if gaze is in front of camera. If it is not, flip direction.
        if gaze_3d[-1] < 0:
            gaze_3d *= -1.0

        g = {
            "eye_center_3d": eye_center.tolist(),
            "gaze_normal_3d": normal_3d.tolist(),
            "gaze_point_3d": gaze_3d.tolist(),
        }

        if self.intrinsics is not None:
            image_point = self.intrinsics.projectPoints(
                gaze_point[np.newaxis], self.rotation_vector, self.translation_vector
            )
            image_point = image_point.reshape(-1, 2)
            image_point = normalize(
                image_point[0], self.intrinsics.resolution, flip_y=True
            )
            image_point = _clamp_norm_point(image_point)
            g["norm_pos"] = image_point

        return g

    def _toWorld(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix, point)[:3]


class Model3D_Binocular(Model3D):
    def __init__(self, *, initial_eye_translation0, initial_eye_translation1, **kwargs):
        super().__init__(**kwargs)
        self._initial_eye_translation0 = np.array(initial_eye_translation0)
        self._initial_eye_translation1 = np.array(initial_eye_translation1)

    def _fit(self, X, Y):
        assert X.shape[1] == _BINOCULAR_FEATURE_COUNT, X
        unprojected_ref_points = Y

        eyeid_left = X[:, _MONOCULAR_EYEID]
        sphere_pos1 = X[-1, _MONOCULAR_SPHERE_CENTER]  # last pupil sphere center
        pupil1_normals = X[:, _MONOCULAR_PUPIL_NORMAL]

        eyeid_right = X[:, _BINOCULAR_EYEID]
        sphere_pos0 = X[-1, _BINOCULAR_SPHERE_CENTER]  # last pupil sphere center
        pupil0_normals = X[:, _BINOCULAR_PUPIL_NORMAL]

        assert (eyeid_left == 1).all(), eyeid_left
        assert (eyeid_right == 0).all(), eyeid_right
        assert sphere_pos1.shape == (3,), sphere_pos1
        assert sphere_pos0.shape == (3,), sphere_pos0

        res = calibrate_binocular(
            unprojected_ref_points,
            pupil0_normals,
            pupil1_normals,
            self.initial_depth,
            initial_translation0=self._initial_eye_translation0,
            initial_translation1=self._initial_eye_translation1,
        )
        success, poses_in_world, gaze_targets_in_world = res
        if not success:
            raise FitDidNotConvergeError

        world_pose, eye0_pose, eye1_pose = poses_in_world

        eye0_cam_pose_in_world = get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
        eye1_cam_pose_in_world = get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

        all_observations = [unprojected_ref_points, pupil0_normals, pupil1_normals]
        nearest_points = calculate_nearest_points_to_targets(
            all_observations, poses_in_world, gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

        params = {
            "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
            "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
        }
        return params

    def set_params(self, **params):
        super().set_params(**params)
        self.last_gaze_distance = 500.0

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

    def _predict_single(self, x):
        assert x.ndim == 1, x
        assert x.shape[0] == _BINOCULAR_FEATURE_COUNT, x
        # find the nearest intersection point of the two gaze lines
        # eye ball centers in world coords
        s1_center = self._eye1_to_World(x[_MONOCULAR_SPHERE_CENTER])
        s0_center = self._eye0_to_World(x[_BINOCULAR_SPHERE_CENTER])
        # eye line of sight in world coords
        s1_normal = np.dot(self.rotation_matricies[1], x[_MONOCULAR_PUPIL_NORMAL])
        s0_normal = np.dot(self.rotation_matricies[0], x[_BINOCULAR_PUPIL_NORMAL])

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

        if nearest_intersection_point is None:
            return None

        # Check if gaze is in front of camera. If it is not, flip direction.
        if nearest_intersection_point[-1] < 0:
            nearest_intersection_point *= -1.0

        g = {
            "eye_centers_3d": {"0": s0_center.tolist(), "1": s1_center.tolist()},
            "gaze_normals_3d": {"0": s0_normal.tolist(), "1": s1_normal.tolist()},
            "gaze_point_3d": nearest_intersection_point.tolist(),
        }

        if self.intrinsics is not None:
            cyclop_gaze = nearest_intersection_point - cyclop_center
            self.last_gaze_distance = np.sqrt(cyclop_gaze.dot(cyclop_gaze))
            image_point = self.intrinsics.projectPoints(
                np.array([nearest_intersection_point])
            )
            image_point = image_point.reshape(-1, 2)
            image_point = normalize(
                image_point[0], self.intrinsics.resolution, flip_y=True
            )
            image_point = _clamp_norm_point(image_point)
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


class Gazer3D(GazerBase):
    label = "3D"

    eye0_hardcoded_translation = 20, 15, -20
    eye1_hardcoded_translation = -40, 15, -20
    ref_depth_hardcoded = 500

    @classmethod
    def _gazer_description_text(cls) -> str:
        return "3D gaze mapping: default method; able to compensate for small movements of the headset (slippage); uses 3d eye model as input."

    def _init_left_model(self) -> Model:
        return Model3D_Monocular(
            intrinsics=self.g_pool.capture.intrinsics,
            initial_depth=self.ref_depth_hardcoded,
            initial_eye_translation=self.eye1_hardcoded_translation,
        )

    def _init_right_model(self) -> Model:
        return Model3D_Monocular(
            intrinsics=self.g_pool.capture.intrinsics,
            initial_depth=self.ref_depth_hardcoded,
            initial_eye_translation=self.eye0_hardcoded_translation,
        )

    def _init_binocular_model(self) -> Model:
        return Model3D_Binocular(
            intrinsics=self.g_pool.capture.intrinsics,
            initial_depth=self.ref_depth_hardcoded,
            initial_eye_translation0=self.eye0_hardcoded_translation,
            initial_eye_translation1=self.eye1_hardcoded_translation,
        )

    def fit_on_calib_data(self, calib_data):
        super().fit_on_calib_data(calib_data)
        self.left_model.binocular_model = self.binocular_model
        self.right_model.binocular_model = self.binocular_model

    def _extract_pupil_features(self, pupil_data) -> np.ndarray:
        pupil_features = np.array(
            [
                [p["id"], *p["sphere"]["center"], *p["circle_3d"]["normal"]]
                for p in pupil_data
            ]
        )
        # pupil_features[:, _MONOCULAR_EYEID]: eye id
        # pupil_features[:, _MONOCULAR_SPHERE_CENTER]: sphere center x/y/z
        # pupil_features[:, _MONOCULAR_PUPIL_NORMAL]: pupil normal x/y/z
        expected_shaped = len(pupil_data), _MONOCULAR_FEATURE_COUNT
        assert pupil_features.shape == expected_shaped, pupil_features
        return pupil_features

    def _extract_reference_features(self, ref_data) -> np.ndarray:
        ref_2d = np.array([ref["screen_pos"] for ref in ref_data])
        assert ref_2d.shape == (len(ref_data), 2), ref_2d
        ref_3d = self.g_pool.capture.intrinsics.unprojectPoints(ref_2d, normalize=True)
        assert ref_3d.shape == (len(ref_data), 3), ref_3d
        return ref_3d

    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)
            gaze_positions = ...  # Placeholder for gaze_positions

            if num_matched == 2:
                if self.binocular_model.is_fitted:
                    right = self._extract_pupil_features([pupil_match[0]])
                    left = self._extract_pupil_features([pupil_match[1]])
                    X = np.hstack([left, right])
                    gaze_positions = self.binocular_model.predict(X)
                    topic = "gaze.3d.01."
                else:
                    logger.debug(
                        "Prediction failed because binocular model is not fitted"
                    )
            elif num_matched == 1:
                X = self._extract_pupil_features([pupil_match[0]])
                if pupil_match[0]["id"] == 0:
                    if self.right_model.is_fitted:
                        gaze_positions = self.right_model.predict(X)
                        topic = "gaze.3d.0."
                    else:
                        logger.debug(
                            "Prediction failed because right model is not fitted"
                        )
                elif pupil_match[0]["id"] == 1:
                    if self.left_model.is_fitted:
                        gaze_positions = self.left_model.predict(X)
                        topic = "gaze.3d.1."
                    else:
                        logger.debug(
                            "Prediction failed because left model is not fitted"
                        )
            else:
                raise ValueError(
                    f"Unexpected number of matched pupil_data: {num_matched}"
                )

            if gaze_positions is ...:
                continue  # Prediction failed and the reason was logged

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
        # TODO: Use topic to filter
        pupil_data = list(filter(lambda p: "3d" in p["method"], pupil_data))
        pupil_data = super().filter_pupil_data(pupil_data, confidence_threshold)
        return pupil_data
