"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

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
import tasklib.background.patches as bg_patches
from calibration_routines.finish_calibration import select_calibration_method
from gaze_producer import model
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
        calibration.mapping_method,
        g_pool.rec_dir,
        calibration.minimum_confidence,
    )

    args = (fake_gpool, ref_dicts_in_calib_range, pupil_pos_in_calib_range)
    name = "Create calibration {}".format(calibration.name)
    return tasklib.background.create(
        name, _create_calibration, args=args, patches=[bg_patches.IPCLoggingPatch()]
    )


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


def _create_calibration(fake_gpool, ref_dicts_in_calib_range, pupil_pos_in_calib_range):
    method, result = select_calibration_method(
        fake_gpool, pupil_pos_in_calib_range, ref_dicts_in_calib_range
    )

    if result["subject"] == "start_plugin":
        calibration_result = model.CalibrationResult(result["name"], result["args"])
        status = "Calibration successful"
    elif result["subject"] == "calibration.failed":
        logger.error("Calibration failed: {}".format(result["reason"]))
        calibration_result = None
        status = result["reason"]
    else:
        logger.error("Unknown calibration result: {}".format(result))
        calibration_result = None
        status = "Unknown calibration result"
    return status, calibration_result
