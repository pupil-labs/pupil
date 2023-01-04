"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import file_methods as fm
import player_methods as pm
import tasklib
from gaze_mapping import gazer_classes_by_class_name, registered_gazer_classes

from .fake_gpool import FakeGPool

g_pool = None  # set by the plugin


class NotEnoughPupilData(ValueError):
    pass


def create_task(gaze_mapper, calibration):
    assert g_pool, "You forgot to set g_pool by the plugin"
    mapping_window = pm.exact_window(g_pool.timestamps, gaze_mapper.mapping_index_range)
    pupil_pos_in_mapping_range = g_pool.pupil_positions.by_ts_window(mapping_window)
    if not pupil_pos_in_mapping_range:
        raise NotEnoughPupilData

    fake_gpool = FakeGPool.from_g_pool(g_pool)

    # Make a copy of params to ensure there are no mappingproxy instances
    # calibration_params = fm._recursive_deep_copy(calibration.params)
    calibration_params = calibration.params

    args = (
        calibration.gazer_class_name,
        calibration_params,
        fake_gpool,
        pupil_pos_in_mapping_range,
        gaze_mapper.manual_correction_x,
        gaze_mapper.manual_correction_y,
    )
    name = f"Create gaze mapper {gaze_mapper.name}"
    return tasklib.background.create(
        name,
        _map_gaze,
        args=args,
        pass_shared_memory=True,
    )


def _map_gaze(
    gazer_class_name,
    gazer_params,
    fake_gpool,
    pupil_pos_in_mapping_range,
    manual_correction_x,
    manual_correction_y,
    shared_memory,
):
    fake_gpool.import_runtime_plugins()
    gazers_by_name = gazer_classes_by_class_name(registered_gazer_classes())
    gazer_cls = gazers_by_name[gazer_class_name]
    gazer = gazer_cls(fake_gpool, params=gazer_params)

    first_ts = pupil_pos_in_mapping_range[0]["timestamp"]
    last_ts = pupil_pos_in_mapping_range[-1]["timestamp"]
    ts_span = last_ts - first_ts
    curr_ts = first_ts

    for gaze_datum in gazer.map_pupil_to_gaze(pupil_pos_in_mapping_range):
        _apply_manual_correction(gaze_datum, manual_correction_x, manual_correction_y)

        # gazer.map_pupil_to_gaze does not yield gaze with monotonic timestamps.
        # Binocular pupil matches are delayed internally. To avoid non-monotonic
        # progress updates, we use the largest timestamp that has been returned up to
        # the current point in time.
        curr_ts = max(curr_ts, gaze_datum["timestamp"])
        shared_memory.progress = (curr_ts - first_ts) / ts_span

        result = (curr_ts, fm.Serialized_Dict(gaze_datum))
        yield [result]


def _apply_manual_correction(gaze_datum, manual_correction_x, manual_correction_y):
    # ["norm_pos"] is a tuple by default
    gaze_norm_pos = list(gaze_datum["norm_pos"])
    gaze_norm_pos[0] += manual_correction_x
    gaze_norm_pos[1] += manual_correction_y
    gaze_datum["norm_pos"] = gaze_norm_pos
