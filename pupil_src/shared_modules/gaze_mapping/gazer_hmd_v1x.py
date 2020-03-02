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

from gaze_mapping.gazer_base import (
    GazerBase,
    Model,
    data_processing,
    NotEnoughDataError,
)
from .gazer_2d_v1x import Gazer2D_v1x
from .gazer_3d_v1x import Gazer3D_v1x

from calibration_routines.optimization_calibration.calibrate_2d import (
    calibrate_2d_polynomial,
    make_map_function,
)

# TODO: See if any calibration_routines dependency can be removed
from calibration_routines import data_processing
from calibration_routines.finish_calibration import (
    create_converge_error_msg,
    create_not_enough_data_error_msg,
)
from calibration_routines.optimization_calibration import calibration_methods


logger = logging.getLogger(__name__)


class GazerHMD2D_v1x(Gazer2D_v1x):
    label = "HMD 2D (v1)"

    def __init__(
        self,
        g_pool,
        *,
        hmd_video_frame_size,
        outlier_threshold,
        calib_data=None,
        params=None,
    ):
        self.__hmd_video_frame_size = hmd_video_frame_size
        self.__outlier_threshold = outlier_threshold
        super().__init__(g_pool, calib_data=calib_data, params=params)

    # TODO: Implement model fitting based on the legacy code bellow
    # def finish_calibration(self):
    #     pupil_list = self.pupil_list
    #     ref_list = self.ref_list
    #     g_pool = self.g_pool

    #     extracted_data = data_processing.get_data_for_calibration_hmd(
    #         pupil_list, ref_list, mode="2d"
    #     )
    #     if not extracted_data:
    #         self.notify_all(create_not_enough_data_error_msg(g_pool))
    #         return

    #     method, result = calibration_methods.calibrate_2d_hmd(
    #         self.hmd_video_frame_size, *extracted_data
    #     )
    #     if result is None:
    #         self.notify_all(create_converge_error_msg(g_pool))
    #         return

    #     ts = g_pool.get_timestamp()

    #     # Announce success
    #     self.notify_all(
    #         {
    #             "subject": "calibration.successful",
    #             "method": method,
    #             "timestamp": ts,
    #             "record": True,
    #         }
    #     )

    #     # Announce calibration data
    #     self.notify_all(
    #         {
    #             "subject": "calibration.calibration_data",
    #             "timestamp": ts,
    #             "pupil_list": pupil_list,
    #             "ref_list": ref_list,
    #             "calibration_method": method,
    #             "record": True,
    #         }
    #     )

    #     # Start mapper
    #     self.notify_all(result)


class GazerHMD3D_v1x(Gazer3D_v1x):
    label = "HMD 3D (v1)"

    def __init__(self, g_pool, *, eye_translations, calib_data=None, params=None):
        logger.info(f"==> GAZE INIT: {self.__class__.__name__} {__name__}")
        self.__eye_translations = eye_translations
        super().__init__(g_pool, calib_data=calib_data, params=params)

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
