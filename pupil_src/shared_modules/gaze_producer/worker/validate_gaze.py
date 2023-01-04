"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import player_methods as pm
import tasklib
from accuracy_visualizer import Accuracy_Visualizer
from gaze_mapping import gazer_classes_by_class_name, registered_gazer_classes
from methods import normalize

from .fake_gpool import FakeGPool

g_pool = None  # set by the plugin


def create_bg_task(gaze_mapper, calibration, reference_location_storage):
    assert g_pool, "You forgot to set g_pool by the plugin"
    refs_in_validation_range = reference_location_storage.get_in_range(
        gaze_mapper.validation_index_range
    )

    validation_window = pm.exact_window(
        g_pool.timestamps, gaze_mapper.validation_index_range
    )
    pupils_in_validation_range = g_pool.pupil_positions.by_ts_window(validation_window)

    # Make a copy of params to ensure there are no mappingproxy instances
    # calibration_params = fm._recursive_deep_copy(calibration.params)
    calibration_params = calibration.params

    fake_gpool = FakeGPool.from_g_pool(g_pool)

    args = (
        fake_gpool,
        calibration.gazer_class_name,
        calibration_params,
        gaze_mapper,
        pupils_in_validation_range,
        refs_in_validation_range,
    )

    return tasklib.background.create(
        f"validate gaze mapper '{gaze_mapper.name}'",
        validate,
        args=args,
    )


def validate(
    g_pool,
    gazer_class_name,
    gazer_params,
    gaze_mapper,
    pupils_in_validation_range,
    refs_in_validation_range,
):
    g_pool.import_runtime_plugins()
    gazers_by_name = gazer_classes_by_class_name(registered_gazer_classes())
    gazer_class = gazers_by_name[gazer_class_name]

    pupil_list = pupils_in_validation_range
    ref_list = [
        _create_ref_dict(ref, g_pool.capture.frame_size)
        for ref in refs_in_validation_range
    ]

    result = Accuracy_Visualizer.calc_acc_prec_errlines(
        g_pool=g_pool,
        gazer_class=gazer_class,
        gazer_params=gazer_params,
        pupil_list=pupil_list,
        ref_list=ref_list,
        intrinsics=g_pool.capture.intrinsics,
        outlier_threshold=gaze_mapper.validation_outlier_threshold_deg,
    )
    return result.accuracy, result.precision


def _create_ref_dict(ref, frame_size):
    return {
        "screen_pos": ref.screen_pos,
        "norm_pos": normalize(ref.screen_pos, frame_size, flip_y=True),
        "timestamp": ref.timestamp,
    }
