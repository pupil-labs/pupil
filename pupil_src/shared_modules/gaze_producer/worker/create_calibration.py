"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

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
from gaze_mapping import (
    CalibrationError,
    gazer_classes_by_class_name,
    registered_gazer_classes,
)
from gaze_producer import model
from methods import normalize

from .fake_gpool import FakeGPool

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

    fake_gpool = FakeGPool.from_g_pool(g_pool)
    fake_gpool.min_calibration_confidence = calibration.minimum_confidence

    args = (
        fake_gpool,
        calibration.gazer_class_name,
        ref_dicts_in_calib_range,
        pupil_pos_in_calib_range,
    )
    name = f"Create calibration {calibration.name}"
    return tasklib.background.create(name, _create_calibration, args=args)


def _create_ref_dict(ref: model.ReferenceLocation):
    return {
        "screen_pos": ref.screen_pos,
        "norm_pos": normalize(ref.screen_pos, g_pool.capture.frame_size, flip_y=True),
        "timestamp": ref.timestamp,
        "mm_pos": ref.mm_pos,
    }


def _create_calibration(
    fake_gpool, gazer_class_name, ref_dicts_in_calib_range, pupil_pos_in_calib_range
):
    # This is needed to support user-provided gazers
    fake_gpool.import_runtime_plugins()

    gazers_by_name = gazer_classes_by_class_name(registered_gazer_classes())

    try:
        gazer_class = gazers_by_name[gazer_class_name]
    except KeyError:
        logger.debug(
            f"Calibration failed! {gazer_class_name} is not in list of known gazers: "
            f"{list(gazers_by_name.keys())}"
        )
        status = f"Unknown gazer class: {gazer_class_name}"
        calibration_result = None
        return status, calibration_result

    # Deep copy dicts of pupil positions, to pass to the gazer init
    pupil_pos_in_calib_range = [p._deep_copy_dict() for p in pupil_pos_in_calib_range]

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
