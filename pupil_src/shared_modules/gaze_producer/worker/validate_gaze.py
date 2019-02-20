"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import tasklib
import tasklib.background.patches as bg_patches
from accuracy_visualizer import Accuracy_Visualizer
from methods import normalize

file_source = None


def create_bg_task(gaze_mapper, reference_location_storage):
    assert file_source, "You forgot to set capture by the plugin"
    refs_in_validation_range = reference_location_storage.get_in_range(
        gaze_mapper.validation_index_range
    )
    return tasklib.background.create(
        "validate gaze mapper '{}'".format(gaze_mapper.name),
        validate,
        args=(
            gaze_mapper,
            refs_in_validation_range,
            file_source.intrinsics,
            file_source.frame_size,
        ),
    )


def validate(gaze_mapper, refs_in_validation_range, capture_intrinsics, frame_size):
    ref_dicts = [_create_ref_dict(ref, frame_size) for ref in refs_in_validation_range]
    accuracy_result, precision_result, _ = Accuracy_Visualizer.calc_acc_prec_errlines(
        gaze_mapper.gaze,
        ref_dicts,
        capture_intrinsics,
        gaze_mapper.validation_outlier_threshold_deg,
    )
    return accuracy_result, precision_result


def _create_ref_dict(ref, frame_size):
    return {
        "screen_pos": ref.screen_pos,
        "norm_pos": normalize(ref.screen_pos, frame_size, flip_y=True),
        "timestamp": ref.timestamp,
    }
