"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import numpy as np
from head_pose_tracker.function import utils


def calculate(
    camera_intrinsics,
    markers_in_frame,
    marker_id_to_extrinsics,
    camera_extrinsics_prv=None,
    min_n_markers_per_frame=1,
):
    data_for_solvepnp = _prepare_data_for_solvepnp(
        markers_in_frame, marker_id_to_extrinsics, min_n_markers_per_frame
    )
    camera_extrinsics = _calculate(
        camera_intrinsics, data_for_solvepnp, camera_extrinsics_prv
    )
    return camera_extrinsics


def _prepare_data_for_solvepnp(
    markers_in_frame, marker_id_to_extrinsics, min_n_markers_per_frame
):
    # markers_available are the markers which have been known
    # and are detected in this frame.

    markers_available = [
        marker
        for marker in markers_in_frame
        if marker["id"] in marker_id_to_extrinsics.keys()
    ]
    if len(markers_available) < min_n_markers_per_frame:
        return None

    markers_points_3d = [
        utils.convert_marker_extrinsics_to_points_3d(
            marker_id_to_extrinsics[marker["id"]]
        )
        for marker in markers_available
    ]
    markers_points_2d = [marker["verts"] for marker in markers_available]

    markers_points_3d = np.array(markers_points_3d, dtype=np.float32).reshape(-1, 4, 3)
    markers_points_2d = np.array(markers_points_2d, dtype=np.float32).reshape(-1, 4, 2)
    data_for_solvepnp = markers_points_3d, markers_points_2d
    return data_for_solvepnp


def _calculate(camera_intrinsics, data_for_solvepnp, camera_extrinsics_prv):
    if not data_for_solvepnp:
        return None

    markers_points_3d, markers_points_2d = data_for_solvepnp

    retval, rotation, translation = _run_solvepnp(
        camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv
    )
    if _check_result_reasonable(retval, rotation, translation, markers_points_3d):
        camera_extrinsics = utils.merge_extrinsics(rotation, translation)
        return camera_extrinsics

    # if _run_solvepnp with camera_extrinsics_prv could not output reasonable result,
    # then do it again without camera_extrinsics_prv
    retval, rotation, translation = _run_solvepnp(
        camera_intrinsics, markers_points_3d, markers_points_2d
    )
    if _check_result_reasonable(retval, rotation, translation, markers_points_3d):
        camera_extrinsics = utils.merge_extrinsics(rotation, translation)
        return camera_extrinsics
    else:
        return None


def _run_solvepnp(
    camera_intrinsics, markers_points_3d, markers_points_2d, camera_extrinsics_prv=None
):
    assert len(markers_points_3d) == len(markers_points_2d)
    assert markers_points_3d.shape[1:] == (4, 3)
    assert markers_points_2d.shape[1:] == (4, 2)

    if camera_extrinsics_prv is None or np.isnan(camera_extrinsics_prv).any():
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d, markers_points_2d
        )
    else:
        rotation_prv, translation_prv = utils.split_extrinsics(camera_extrinsics_prv)
        retval, rotation, translation = camera_intrinsics.solvePnP(
            markers_points_3d,
            markers_points_2d,
            useExtrinsicGuess=True,
            rvec=rotation_prv.copy(),
            tvec=translation_prv.copy(),
        )
    return retval, rotation, translation


def _check_result_reasonable(retval, rotation, translation, pts_3d_world):
    # solvePnP outputs wrong pose estimations sometimes, so it is necessary to check
    # if the rotation and translation from the output of solvePnP is reasonable.
    if not retval:
        return False

    assert rotation.size == 3 and translation.size == 3

    # the magnitude of rotation should be less than 2*pi
    if (np.abs(rotation) > np.pi * 2).any():
        return False

    # the depth of the markers in the camera coordinate system should be positive,
    # i.e. all seen markers in the frame should be in front of the camera;
    # if not, that implies the output of solvePnP is wrong.
    pts_3d_camera = utils.to_camera_coordinate(pts_3d_world, rotation, translation)
    if (pts_3d_camera[:, 2] < 1).any():
        return False

    return True
