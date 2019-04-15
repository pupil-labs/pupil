"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections

import player_methods as pm
from head_pose_tracker import storage
from head_pose_tracker.function import (
    BundleAdjustment,
    pick_key_markers,
    get_initial_guess,
)
from head_pose_tracker.function import utils

IntrinsicsTuple = collections.namedtuple(
    "IntrinsicsTuple", ["camera_matrix", "dist_coefs"]
)


def optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment):
    try:
        bg_storage.marker_id_to_extrinsics[bg_storage.origin_marker_id]
    except KeyError:
        bg_storage.set_origin_marker_id()

    initial_guess = get_initial_guess.calculate(
        bg_storage.marker_id_to_extrinsics,
        bg_storage.frame_id_to_extrinsics,
        bg_storage.all_key_markers,
        camera_intrinsics,
    )
    if not initial_guess:
        return

    result = bundle_adjustment.calculate(initial_guess)

    marker_id_to_extrinsics = result.marker_id_to_extrinsics
    marker_id_to_points_3d = {
        marker_id: utils.convert_marker_extrinsics_to_points_3d(extrinsics)
        for marker_id, extrinsics in result.marker_id_to_extrinsics.items()
    }
    model_tuple = (
        bg_storage.origin_marker_id,
        marker_id_to_extrinsics,
        marker_id_to_points_3d,
    )
    bg_storage.update_model(*model_tuple)
    intrinsics_tuple = IntrinsicsTuple(camera_intrinsics.K, camera_intrinsics.D)
    return (
        model_tuple,
        result.frame_id_to_extrinsics,
        result.frame_ids_failed,
        intrinsics_tuple,
    )


def offline_optimization(
    timestamps,
    frame_index_range,
    user_defined_origin_marker_id,
    optimize_camera_intrinsics,
    markers_bisector,
    frame_index_to_num_markers,
    camera_intrinsics,
    shared_memory,
):
    def find_markers_in_frame(index):
        window = pm.enclosing_window(timestamps, index)
        return markers_bisector.by_ts_window(window)

    frame_start, frame_end = frame_index_range
    frame_indices_with_marker = [
        frame_index
        for frame_index, num_markers in frame_index_to_num_markers.items()
        if num_markers >= 2
    ]
    frame_indices = list(
        set(range(frame_start, frame_end + 1)) & set(frame_indices_with_marker)
    )
    frame_count = len(frame_indices)

    bg_storage = storage.Markers3DModel(user_defined_origin_marker_id)
    bundle_adjustment = BundleAdjustment(camera_intrinsics, optimize_camera_intrinsics)

    for idx, frame_index in enumerate(frame_indices):
        markers_in_frame = find_markers_in_frame(frame_index)
        bg_storage.all_key_markers += pick_key_markers.run(
            markers_in_frame, bg_storage.all_key_markers
        )

        if not (idx % 50 == 25 or idx == frame_count - 1):
            continue

        shared_memory.progress = (idx + 1) / frame_count
        try:
            (
                model_tuple,
                frame_id_to_extrinsics,
                frame_ids_failed,
                intrinsics_tuple,
            ) = optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment)
        except TypeError:
            pass
        else:
            bg_storage.frame_id_to_extrinsics = frame_id_to_extrinsics
            bg_storage.discard_failed_key_markers(frame_ids_failed)

            yield model_tuple, intrinsics_tuple


def online_optimization(
    origin_marker_id,
    marker_id_to_extrinsics_opt,
    frame_id_to_extrinsics_opt,
    all_key_markers,
    optimize_camera_intrinsics,
    camera_intrinsics,
):
    bg_storage = storage.Markers3DModel()
    bg_storage.origin_marker_id = origin_marker_id
    bg_storage.marker_id_to_extrinsics = marker_id_to_extrinsics_opt
    bg_storage.frame_id_to_extrinsics = frame_id_to_extrinsics_opt
    bg_storage.all_key_markers = all_key_markers

    bundle_adjustment = BundleAdjustment(camera_intrinsics, optimize_camera_intrinsics)

    return optimization_routine(bg_storage, camera_intrinsics, bundle_adjustment)
