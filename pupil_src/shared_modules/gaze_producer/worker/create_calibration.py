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
from time import time
from types import SimpleNamespace

import player_methods as pm
import tasklib.background
from calibration_routines.finish_calibration import (
    select_method_and_perform_calibration,
)
from gaze_producer import model
from gaze_mapping import registered_gazer_classes, CalibrationError
from methods import normalize

logger = logging.getLogger(__name__)

g_pool = None  # set by the plugin


def create_task(calibration, all_reference_locations):
    assert g_pool, "You forgot to set g_pool by the plugin"
    calibration_window = pm.exact_window(
        g_pool.timestamps, calibration.frame_index_range
    )
    pupil_pos_in_calib_range = g_pool.pupil_positions.by_ts_window(calibration_window)

    frame_start = calibration.frame_index_range[0]
    frame_end = calibration.frame_index_range[1]
    ref_dicts_in_calib_range = [
        _create_ref_dict(ref)
        for ref in all_reference_locations
        if frame_start <= ref.frame_index <= frame_end
    ]

    fake_gpool = _setup_fake_gpool(
        g_pool.capture.frame_size,
        g_pool.capture.intrinsics,
        g_pool.rec_dir,
        calibration.minimum_confidence,
    )

    args = (
        fake_gpool,
        calibration.gazer_class_name,
        ref_dicts_in_calib_range,
        pupil_pos_in_calib_range,
    )
    name = "Create calibration {}".format(calibration.name)
    return tasklib.background.create(name, _create_calibration, args=args)


def _create_ref_dict(ref):
    return {
        "screen_pos": ref.screen_pos,
        "norm_pos": normalize(ref.screen_pos, g_pool.capture.frame_size, flip_y=True),
        "timestamp": ref.timestamp,
    }


def _setup_fake_gpool(
    frame_size, intrinsics, detection_mapping_mode, rec_dir, min_calibration_confidence
):
    cap = SimpleNamespace()
    cap.frame_size = frame_size
    cap.intrinsics = intrinsics
    pool = SimpleNamespace()
    pool.capture = cap
    pool.get_timestamp = time
    pool.detection_mapping_mode = detection_mapping_mode
    pool.min_calibration_confidence = min_calibration_confidence
    pool.rec_dir = rec_dir
    pool.app = "player"
    return pool


def _create_calibration(
    fake_gpool, gazer_class_name, ref_dicts_in_calib_range, pupil_pos_in_calib_range
):
    registered_gazer_classes_by_name = {
        cls.__name__: cls for cls in registered_gazer_classes
    }
    try:
        gazer_class = registered_gazer_classes_by_name[gazer_class_name]
    except KeyError:
        logger.debug(
            f"Calibration failed! {gazer_class_name} is not in list of known gazers: "
            f"{list(registered_gazer_classes_by_name.keys())}"
        )
        status = "Unknown mapping method"
        calibration_result = None
        return status, calibration_result

    try:
        calib_data = {
            "ref_list": ref_dicts_in_calib_range,
            "pupil_list": pupil_pos_in_calib_range,
        }
        gazer = gazer_class(
            fake_gpool, calib_data=calib_data, raise_calibration_error=True
        )
        calibration_result = model.CalibrationResult(
            gazer_class_name, gazer.get_params()
        )
        status = "Calibration successful"
        return status, calibration_result

    except CalibrationError as err:
        from traceback import format_exc

        logger.debug(f"Calibration failed! Traceback:\n{format_exc()}")
        status = f"Calibration failed: {err.message}"
        calibration_result = None
        return status, calibration_result
