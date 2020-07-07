"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T
import numpy as np

from methods import normalize

from .calibrate_3d import calibrate_hmd
from .utils import (
    calculate_nearest_points_to_targets,
    get_eye_cam_pose_in_world,
)

from gaze_mapping.gazer_base import (
    GazerBase,
    Model,
    CalibrationError,
    NotEnoughDataError,
    FitDidNotConvergeError,
)
from .gazer_headset import Gazer3D, Model3D_Binocular, Model3D_Monocular

from .utils import _clamp_norm_point

_REFERENCE_FEATURE_COUNT = 3

_MONOCULAR_FEATURE_COUNT = 7
_MONOCULAR_EYEID = 0
_MONOCULAR_SPHERE_CENTER = slice(1, 4)
_MONOCULAR_PUPIL_NORMAL = slice(4, 7)

_BINOCULAR_FEATURE_COUNT = 14
_BINOCULAR_EYEID = 7
_BINOCULAR_SPHERE_CENTER = slice(8, 11)
_BINOCULAR_PUPIL_NORMAL = slice(11, 14)


logger = logging.getLogger(__name__)


class MissingEyeTranslationsError(CalibrationError):
    message = (
        "GazerHMD3D can only be calibrated if it is "
        "initialised with valid eye translations."
    )


class ModelHMD3D_Binocular(Model3D_Binocular):
    def __init__(self, *, intrinsics, eye_translations):
        self.intrinsics = intrinsics
        self.eye_translations = eye_translations
        self._is_fitted = False

    def _fit(self, X, Y):
        if self.eye_translations is None:
            raise MissingEyeTranslationsError()
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

        res = calibrate_hmd(
            unprojected_ref_points,
            pupil0_normals,
            pupil1_normals,
            self.eye_translations,
        )
        success, poses_in_world, gaze_targets_in_world = res
        if not success:
            raise FitDidNotConvergeError

        eye0_pose, eye1_pose = poses_in_world

        eye0_cam_pose_in_world = get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
        eye1_cam_pose_in_world = get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

        all_observations = [unprojected_ref_points, pupil0_normals, pupil1_normals]
        nearest_points = calculate_nearest_points_to_targets(
            all_observations, [np.zeros(6), *poses_in_world], gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

        params = {
            "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
            "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
        }
        return params


class ModelHMD3D_Monocular(Model3D_Monocular):
    def _fit(self, X, Y):
        return NotImplemented


class GazerHMD3D(Gazer3D):
    label = "HMD 3D"

    @classmethod
    def _gazer_description_text(cls) -> str:
        return "Gaze mapping built specifically for HMD-Eyes."

    def __init__(self, g_pool, *, eye_translations=None, calib_data=None, params=None):
        self.__eye_translations = eye_translations
        super().__init__(g_pool, calib_data=calib_data, params=params)

    @property
    def _gpool_capture_intrinsics_if_available(self) -> T.Optional[T.Any]:
        if hasattr(self.g_pool, "capture"):
            return self.g_pool.capture.intrinsics
        else:
            return None

    def _init_binocular_model(self) -> Model:
        return ModelHMD3D_Binocular(
            intrinsics=self._gpool_capture_intrinsics_if_available,
            eye_translations=self.__eye_translations,
        )

    def _init_left_model(self) -> Model:
        return ModelHMD3D_Monocular(
            intrinsics=self._gpool_capture_intrinsics_if_available
        )

    def _init_right_model(self) -> Model:
        return ModelHMD3D_Monocular(
            intrinsics=self._gpool_capture_intrinsics_if_available
        )

    def fit_on_calib_data(self, calib_data):
        # extract reference data
        ref_data = calib_data["ref_list"]
        # extract and filter pupil data
        pupil_data = calib_data["pupil_list"]
        pupil_data = self.filter_pupil_data(
            pupil_data, self.g_pool.min_calibration_confidence
        )
        # match pupil to reference data (left, right, and binocular)
        matches = self.match_pupil_to_ref(pupil_data, ref_data)
        if matches.binocular[0]:
            self._fit_binocular_model(self.binocular_model, matches.binocular)
            params = self.binocular_model.get_params()
            self.left_model.set_params(
                eye_camera_to_world_matrix=params["eye_camera_to_world_matrix1"],
                gaze_distance=self.binocular_model.last_gaze_distance,
            )
            self.right_model.set_params(
                eye_camera_to_world_matrix=params["eye_camera_to_world_matrix0"],
                gaze_distance=self.binocular_model.last_gaze_distance,
            )
            self.left_model.binocular_model = self.binocular_model
            self.right_model.binocular_model = self.binocular_model
        else:
            raise NotEnoughDataError

    def _extract_reference_features(self, ref_data) -> np.ndarray:
        ref_3d = np.array([ref["mm_pos"] for ref in ref_data])
        assert ref_3d.shape == (len(ref_data), 3), ref_3d
        return ref_3d

    def get_init_dict(self):
        return {
            **super().get_init_dict(),
            "eye_translations": self.__eye_translations,
        }
