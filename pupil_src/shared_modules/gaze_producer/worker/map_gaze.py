"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from time import time

import file_methods as fm
import player_methods as pm
import tasklib
import tasklib.background.patches as bg_patches
from calibration_routines import gaze_mapping_plugins
from types import SimpleNamespace

g_pool = None  # set by the plugin


def create_task(gaze_mapper, calibration):
    assert g_pool, "You forgot to set g_pool by the plugin"
    mapping_window = pm.exact_window(g_pool.timestamps, gaze_mapper.mapping_index_range)
    pupil_pos_in_mapping_range = g_pool.pupil_positions.by_ts_window(mapping_window)

    fake_gpool = _setup_fake_gpool(
        g_pool.capture.frame_size,
        g_pool.capture.intrinsics,
        calibration.mapping_method,
        g_pool.rec_dir,
    )

    args = (
        calibration.result,
        fake_gpool,
        pupil_pos_in_mapping_range,
        gaze_mapper.manual_correction_x,
        gaze_mapper.manual_correction_y,
    )
    name = "Create gaze mapper {}".format(gaze_mapper.name)
    return tasklib.background.create(
        name,
        _map_gaze,
        args=args,
        patches=[bg_patches.IPCLoggingPatch()],
        pass_shared_memory=True,
    )


def _setup_fake_gpool(frame_size, intrinsics, detection_mapping_mode, rec_dir):
    cap = SimpleNamespace()
    cap.frame_size = frame_size
    cap.intrinsics = intrinsics
    pool = SimpleNamespace()
    pool.capture = cap
    pool.get_timestamp = time
    pool.detection_mapping_mode = detection_mapping_mode
    pool.rec_dir = rec_dir
    pool.app = "player"
    return pool


def _map_gaze(
    calibration_result,
    fake_gpool,
    pupil_pos_in_mapping_range,
    manual_correction_x,
    manual_correction_y,
    shared_memory,
):
    gaze_mapping_plugins_by_name = {p.__name__: p for p in gaze_mapping_plugins}
    gaze_mapper_cls = gaze_mapping_plugins_by_name[
        calibration_result.mapping_plugin_name
    ]
    gaze_mapper = gaze_mapper_cls(fake_gpool, **calibration_result.mapper_args)

    for idx_incoming, pupil_pos in enumerate(pupil_pos_in_mapping_range):
        mapped_gaze = gaze_mapper.on_pupil_datum(pupil_pos)

        output_gaze = []
        for gaze_datum in mapped_gaze:
            _apply_manual_correction(
                gaze_datum, manual_correction_x, manual_correction_y
            )
            output_gaze.append(
                (gaze_datum["timestamp"], fm.Serialized_Dict(gaze_datum))
            )

        shared_memory.progress = (idx_incoming + 1) / len(pupil_pos_in_mapping_range)

        if output_gaze:
            yield output_gaze


def _apply_manual_correction(gaze_datum, manual_correction_x, manual_correction_y):
    # ["norm_pos"] is a tuple by default
    gaze_norm_pos = list(gaze_datum["norm_pos"])
    gaze_norm_pos[0] += manual_correction_x
    gaze_norm_pos[1] += manual_correction_y
    gaze_datum["norm_pos"] = gaze_norm_pos
