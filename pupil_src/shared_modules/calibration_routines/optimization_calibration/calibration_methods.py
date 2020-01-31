"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np

from calibration_routines.optimization_calibration import (
    calibrate_3d,
    calibrate_2d,
    utils,
)


def calibrate_3d_binocular(
    unprojected_ref_points,
    pupil0_normals,
    pupil1_normals,
    last_pupil0,
    last_pupil1,
    initial_depth=500,
):
    method = "binocular 3d model"

    res = calibrate_3d.calibrate_binocular(
        unprojected_ref_points, pupil0_normals, pupil1_normals, initial_depth
    )
    success, poses_in_world, gaze_targets_in_world = res
    if not success:
        return method, None

    world_pose, eye0_pose, eye1_pose = poses_in_world

    sphere_pos0 = last_pupil0["sphere"]["center"]
    sphere_pos1 = last_pupil1["sphere"]["center"]
    eye0_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
    eye1_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

    all_observations = [unprojected_ref_points, pupil0_normals, pupil1_normals]
    nearest_points = utils.calculate_nearest_points_to_targets(
        all_observations, poses_in_world, gaze_targets_in_world
    )
    nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

    mapper = "Binocular_Vector_Gaze_Mapper"
    args = {
        "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
        "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
        "cal_points_3d": gaze_targets_in_world.tolist(),
        "cal_ref_points_3d": nearest_points_world.tolist(),
        "cal_gaze_points0_3d": nearest_points_eye0.tolist(),
        "cal_gaze_points1_3d": nearest_points_eye1.tolist(),
    }
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_3d_monocular(
    unprojected_ref_points, pupil_normals, last_pupil, initial_depth=500
):
    method = "monocular 3d model"

    pupil_id = last_pupil["id"]
    res = calibrate_3d.calibrate_monocular(
        unprojected_ref_points, pupil_normals, pupil_id, initial_depth
    )
    success, poses_in_world, gaze_targets_in_world = res
    if not success:
        return method, None

    world_pose, eye_pose = poses_in_world

    sphere_pos = last_pupil["sphere"]["center"]
    eye_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye_pose, sphere_pos)

    all_observations = [unprojected_ref_points, pupil_normals]
    nearest_points = utils.calculate_nearest_points_to_targets(
        all_observations, poses_in_world, gaze_targets_in_world
    )
    nearest_points_world, nearest_points_eye = nearest_points

    mapper = "Vector_Gaze_Mapper"
    args = {
        "eye_camera_to_world_matrix": eye_cam_pose_in_world.tolist(),
        "cal_points_3d": gaze_targets_in_world.tolist(),
        "cal_ref_points_3d": nearest_points_world.tolist(),
        "cal_gaze_points_3d": nearest_points_eye.tolist(),
        "gaze_distance": initial_depth,
    }
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_2d_binocular(
    g_pool, cal_pt_cloud_binocular, cal_pt_cloud0, cal_pt_cloud1
):
    method = "binocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud_binocular, g_pool.capture.frame_size, binocular=True
    )
    if not inliers.any():
        return method, None

    map_fn, inliers, params_eye0 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud0, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    map_fn, inliers, params_eye1 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud1, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Binocular_Gaze_Mapper"
    args = {"params": params, "params_eye0": params_eye0, "params_eye1": params_eye1}
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_2d_monocular(g_pool, cal_pt_cloud):
    method = "monocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Monocular_Gaze_Mapper"
    args = {"params": params}
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_3d_hmd(
    ref_points_3d,
    pupil0_normals,
    pupil1_normals,
    last_pupil0,
    last_pupil1,
    eye_translations,
):
    method = "hmd binocular 3d model"

    res = calibrate_3d.calibrate_hmd(
        ref_points_3d, pupil0_normals, pupil1_normals, eye_translations
    )
    success, poses_in_world, gaze_targets_in_world = res
    if not success:
        return method, None

    eye0_pose, eye1_pose = poses_in_world

    sphere_pos0 = last_pupil0["sphere"]["center"]
    sphere_pos1 = last_pupil1["sphere"]["center"]
    eye0_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
    eye1_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

    all_observations = [gaze_targets_in_world, pupil0_normals, pupil1_normals]
    nearest_points = utils.calculate_nearest_points_to_targets(
        all_observations, [np.zeros(6), *poses_in_world], gaze_targets_in_world
    )
    nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

    mapper = "Binocular_Vector_Gaze_Mapper"
    args = {
        "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
        "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
        "cal_points_3d": gaze_targets_in_world.tolist(),
        "cal_ref_points_3d": nearest_points_world.tolist(),
        "cal_gaze_points0_3d": nearest_points_eye0.tolist(),
        "cal_gaze_points1_3d": nearest_points_eye1.tolist(),
    }
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_2d_hmd(hmd_video_frame_size, cal_pt_cloud0, cal_pt_cloud1):
    params0, params1 = None, None

    if cal_pt_cloud0:
        map_fn0, inliers0, params0 = calibrate_2d.calibrate_2d_polynomial(
            cal_pt_cloud0, hmd_video_frame_size, binocular=False
        )
        if not inliers0.any():
            return None, None
    if cal_pt_cloud1:
        map_fn1, inliers1, params1 = calibrate_2d.calibrate_2d_polynomial(
            cal_pt_cloud1, hmd_video_frame_size, binocular=False
        )
        if not inliers1.any():
            return None, None

    if params0 and params1:
        method = "dual monocular polynomial regression"
        mapper = "Dual_Monocular_Gaze_Mapper"
        args = {"params0": params0, "params1": params1}
    elif params0 or params1:
        method = "monocular polynomial regression"
        mapper = "Monocular_Gaze_Mapper"
        args = {"params": params0 if params0 else params1}
    else:
        # This case should not happen.
        # If cal_pt_cloud0 and cal_pt_cloud1 are both empty lists,
        # not_enough_data_error_msg should have been shown.
        raise RuntimeError

    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result
