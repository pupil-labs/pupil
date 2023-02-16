"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections

import numpy as np
from head_pose_tracker.function import solvepnp, triangulate_marker

InitialGuess = collections.namedtuple(
    "InitialGuess", ["key_markers", "frame_id_to_extrinsics", "marker_id_to_extrinsics"]
)


def calculate(
    marker_id_to_extrinsics_opt,
    frame_id_to_extrinsics_opt,
    key_markers,
    camera_intrinsics,
):
    """get marker and camera initial guess for bundle adjustment"""

    marker_id_to_extrinsics_init = {
        marker_id: np.array(extrinsics)
        for marker_id, extrinsics in marker_id_to_extrinsics_opt.items()
    }
    frame_id_to_extrinsics_init = {
        frame_id: np.array(extrinsics)
        for frame_id, extrinsics in frame_id_to_extrinsics_opt.items()
    }
    frame_ids = list({marker.frame_id for marker in key_markers})
    marker_ids = list({marker.marker_id for marker in key_markers})

    # The function _calculate_extrinsics calculates camera extrinsics and marker
    # extrinsics iteratively. It is possible that not all of them can be calculated
    # after one run of _calculate_extrinsics, so we need to run it twice.
    for _ in range(2):
        frame_id_to_extrinsics_init = _get_frame_id_to_extrinsics_init(
            camera_intrinsics,
            key_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
            frame_ids,
        )
        marker_id_to_extrinsics_init = _get_marker_id_to_extrinsics_init(
            camera_intrinsics,
            key_markers,
            frame_id_to_extrinsics_init,
            marker_id_to_extrinsics_init,
            marker_ids,
        )

    key_markers_useful = [
        key_marker
        for key_marker in key_markers
        if (
            key_marker.frame_id in frame_id_to_extrinsics_init.keys()
            and key_marker.marker_id in marker_id_to_extrinsics_init.keys()
        )
    ]
    if not key_markers_useful:
        return None

    initial_guess = InitialGuess(
        key_markers_useful, frame_id_to_extrinsics_init, marker_id_to_extrinsics_init
    )
    return initial_guess


def _get_frame_id_to_extrinsics_init(
    camera_intrinsics,
    key_markers,
    frame_id_to_extrinsics_init,
    marker_id_to_extrinsics_init,
    frame_ids,
):
    """calculate camera extrinsics based on the known marker extrinsics"""

    frame_ids_not_computed = set(frame_ids) - set(frame_id_to_extrinsics_init.keys())
    for frame_id in frame_ids_not_computed:
        markers_in_frame = [
            {"id": marker.marker_id, "verts": marker.verts}
            for marker in key_markers
            if marker.frame_id == frame_id
            and marker.marker_id in marker_id_to_extrinsics_init.keys()
        ]

        camera_extrinsics = solvepnp.calculate(
            camera_intrinsics, markers_in_frame, marker_id_to_extrinsics_init
        )
        if camera_extrinsics is not None:
            frame_id_to_extrinsics_init[frame_id] = camera_extrinsics

    return frame_id_to_extrinsics_init


def _get_marker_id_to_extrinsics_init(
    camera_intrinsics,
    key_markers,
    frame_id_to_extrinsics_init,
    marker_id_to_extrinsics_init,
    marker_ids,
):
    """calculate marker extrinsics based on the known camera extrinsics"""

    marker_ids_not_computed = set(marker_ids) - set(marker_id_to_extrinsics_init.keys())
    for marker_id in marker_ids_not_computed:
        frame_id_to_detections = {
            marker.frame_id: {"verts": marker.verts}
            for marker in key_markers
            if marker.marker_id == marker_id
            and marker.frame_id in frame_id_to_extrinsics_init
        }

        marker_extrinsics = triangulate_marker.calculate(
            camera_intrinsics, frame_id_to_detections, frame_id_to_extrinsics_init
        )
        if marker_extrinsics is not None:
            marker_id_to_extrinsics_init[marker_id] = marker_extrinsics

    return marker_id_to_extrinsics_init
