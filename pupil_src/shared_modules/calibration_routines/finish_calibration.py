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
import os

import file_methods as fm
from calibration_routines import data_processing
from calibration_routines.optimization_calibration import calibration_methods

logger = logging.getLogger(__name__)


def finish_calibration(g_pool, pupil_list, ref_list):
    method, result = select_method_and_perform_calibration(g_pool, pupil_list, ref_list)

    # Start mapper / announce error
    g_pool.active_calibration_plugin.notify_all(result)
    if result["subject"] == "calibration.failed":
        return

    ts = g_pool.get_timestamp()

    # Announce success
    g_pool.active_calibration_plugin.notify_all(
        {
            "subject": "calibration.successful",
            "method": method,
            "timestamp": ts,
            "record": True,
        }
    )

    # Announce calibration data
    user_calibration_data = {
        "timestamp": ts,
        "pupil_list": pupil_list,
        "ref_list": ref_list,
        "calibration_method": method,
        "mapper_name": result["name"],
        "mapper_args": result["args"],
    }
    fm.save_object(
        user_calibration_data, os.path.join(g_pool.user_dir, "user_calibration_data")
    )
    g_pool.active_calibration_plugin.notify_all(
        {
            "subject": "calibration.calibration_data",
            "record": True,
            **user_calibration_data,
        }
    )


def select_method_and_perform_calibration(g_pool, pupil_list, ref_list):
    mode = g_pool.detection_mapping_mode
    if mode == "3d" and not (
        hasattr(g_pool.capture, "intrinsics") or g_pool.capture.intrinsics
    ):
        mode = "2d"
        logger.warning(
            "Please calibrate your world camera using 'camera intrinsics estimation' "
            "for 3d gaze mapping."
        )

    binocular, extracted_data = data_processing.get_data_for_calibration(
        g_pool, pupil_list, ref_list, mode
    )

    if not extracted_data:
        return None, create_not_enough_data_error_msg(g_pool)

    if mode == "3d" and binocular:
        method, result = calibration_methods.calibrate_3d_binocular(*extracted_data)
    elif mode == "3d" and not binocular:
        method, result = calibration_methods.calibrate_3d_monocular(*extracted_data)
    elif mode == "2d" and binocular:
        method, result = calibration_methods.calibrate_2d_binocular(
            g_pool, *extracted_data
        )
    elif mode == "2d" and not binocular:
        method, result = calibration_methods.calibrate_2d_monocular(
            g_pool, *extracted_data
        )
    else:
        raise RuntimeError("This case should not happen.")

    if result is None:
        return method, create_converge_error_msg(g_pool)

    return method, result


def create_not_enough_data_error_msg(g_pool):
    msg = "Not enough ref points or pupil data available for calibration."
    logger.error(msg)
    return {
        "subject": "calibration.failed",
        "reason": msg,
        "timestamp": g_pool.get_timestamp(),
        "record": True,
    }


def create_converge_error_msg(g_pool):
    msg = "Parameters could not be estimated from data."
    logger.error(msg)
    return {
        "subject": "calibration.failed",
        "reason": msg,
        "timestamp": g_pool.get_timestamp(),
        "record": True,
    }
