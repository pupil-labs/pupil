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

# TODO: See if any calibration_routines dependency can be removed
from calibration_routines import data_processing
from calibration_routines.finish_calibration import create_converge_error_msg
from calibration_routines.finish_calibration import create_not_enough_data_error_msg
from calibration_routines.gaze_mappers import _clamp_norm_point, normalize
from calibration_routines.optimization_calibration import calibration_methods
from calibration_routines.optimization_calibration import utils
from calibration_routines.optimization_calibration.calibrate_3d import calibrate_hmd

from gaze_mapping.gazer_base import (
    GazerBase,
    Model,
    NotEnoughDataError,
    FitDidNotConvergeError,
)
from .gazer_3d_v1x import Gazer3D_v1x, Model3D_v1x_Binocular


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


class ModelHMD3D_v1x(Model3D_v1x_Binocular):
    def __init__(self, *, intrinsics, eye_translations):
        self.intrinsics = intrinsics
        self.eye_translations = eye_translations
        self._is_fitted = False

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

        eye0_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
        eye1_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

        all_observations = [unprojected_ref_points, pupil0_normals, pupil1_normals]
        nearest_points = utils.calculate_nearest_points_to_targets(
            all_observations, [np.zeros(6), *poses_in_world], gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

        params = {
            "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
            "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
        }
        return params


class GazerHMD3D_v1x(Gazer3D_v1x):
    label = "HMD 3D (v1)"

    def __init__(self, g_pool, *, eye_translations, calib_data=None, params=None):
        self.__eye_translations = eye_translations
        super().__init__(g_pool, calib_data=calib_data, params=params)

    def _init_binocular_model(self) -> Model:
        return ModelHMD3D_v1x(
            intrinsics=self.g_pool.capture.intrinsics,
            eye_translations=self.__eye_translations,
        )

    def _init_left_model(self) -> Model:
        return ModelHMD3D_v1x(
            intrinsics=self.g_pool.capture.intrinsics,
            eye_translations=self.__eye_translations,
        )

    def _init_right_model(self) -> Model:
        return ModelHMD3D_v1x(
            intrinsics=self.g_pool.capture.intrinsics,
            eye_translations=self.__eye_translations,
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
            self.left_model.set_params(**params)
            self.right_model.set_params(**params)
        else:
            raise NotEnoughDataError

    def _extract_reference_features(self, ref_data) -> np.ndarray:
        ref_3d = np.array([ref["mm_pos"] for ref in ref_data])
        assert ref_3d.shape == (len(ref_data), 3), ref_3d
        return ref_3d

    # TODO: Implement model fitting based on the legacy code bellow
    # def finish_calibration(self):
    #     pupil_list = self.pupil_list
    #     ref_list = self.ref_list
    #     g_pool = self.g_pool

    #     extracted_data = data_processing.get_data_for_calibration_hmd(
    #         pupil_list, ref_list, mode="3d"
    #     )
    #     if not extracted_data:
    #         self.notify_all(create_not_enough_data_error_msg(g_pool))
    #         return

    #     method, result = calibration_methods.calibrate_3d_hmd(
    #         *extracted_data, self.eye_translations
    #     )
    #     if result is None:
    #         self.notify_all(create_converge_error_msg(g_pool))
    #         return

    #     ts = g_pool.get_timestamp()

    #     # Announce success
    #     g_pool.active_calibration_plugin.notify_all(
    #         {
    #             "subject": "calibration.successful",
    #             "method": method,
    #             "timestamp": ts,
    #             "record": True,
    #         }
    #     )

    #     # Announce calibration data
    #     # this is only used by show calibration. TODO: rewrite show calibration.
    #     user_calibration_data = {
    #         "timestamp": ts,
    #         "pupil_list": pupil_list,
    #         "ref_list": ref_list,
    #         "calibration_method": method,
    #     }
    #     fm.save_object(
    #         user_calibration_data,
    #         os.path.join(g_pool.user_dir, "user_calibration_data"),
    #     )
    #     g_pool.active_calibration_plugin.notify_all(
    #         {
    #             "subject": "calibration.calibration_data",
    #             "record": True,
    #             **user_calibration_data,
    #         }
    #     )

    #     # Start mapper
    #     result["args"]["backproject"] = hasattr(g_pool, "capture")
    #     self.g_pool.active_calibration_plugin.notify_all(result)
