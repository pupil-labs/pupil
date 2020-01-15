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

import cv2
import numpy as np

from calibration_routines import calibrate
from calibration_routines.optimization_calibration import utils, BundleAdjustment
from file_methods import save_object

logger = logging.getLogger(__name__)

not_enough_data_error_msg = (
    "Not enough ref points or pupil data available for calibration."
)
solver_failed_to_converge_error_msg = "Parameters could not be estimated from data."

eye0_hardcoded_translation = np.array([20, 15, -20])
eye1_hardcoded_translation = np.array([-40, 15, -20])


class SphericalCamera:
    def __init__(
        self, observations, rotation, translation, fix_rotation, fix_translation
    ):
        self.observations = observations
        self.rotation = rotation
        self.translation = translation
        self.fix_rotation = bool(fix_rotation)
        self.fix_translation = bool(fix_translation)


def calibrate_3d_binocular(
    g_pool, matched_binocular_data, pupil0, pupil1, initial_depth=500
):
    method = "binocular 3d model"

    unprojected_ref_points, pupil0_normals, pupil1_normals = calibrate.preprocess_3d_data(
        matched_binocular_data, g_pool
    )
    if (
        len(unprojected_ref_points) < 1
        or len(pupil0_normals) < 1
        or len(pupil1_normals) < 1
    ):
        logger.error(not_enough_data_error_msg)
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": not_enough_data_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    unprojected_ref_points = np.asarray(unprojected_ref_points)
    pupil0_normals = np.asarray(pupil0_normals)
    pupil1_normals = np.asarray(pupil1_normals)

    # initial_rotation and initial_translation are eye pose in world coordinates
    initial_rotation0 = utils.get_initial_eye_camera_rotation(
        pupil0_normals, unprojected_ref_points
    )
    initial_rotation1 = utils.get_initial_eye_camera_rotation(
        pupil1_normals, unprojected_ref_points
    )
    initial_translation0 = eye0_hardcoded_translation
    initial_translation1 = eye1_hardcoded_translation

    world = SphericalCamera(
        observations=unprojected_ref_points,
        rotation=np.zeros(3),
        translation=np.zeros(3),
        fix_rotation=True,
        fix_translation=True,
    )
    eye0 = SphericalCamera(
        observations=pupil0_normals,
        rotation=initial_rotation0,
        translation=initial_translation0,
        fix_rotation=False,
        fix_translation=True,
    )
    eye1 = SphericalCamera(
        observations=pupil1_normals,
        rotation=initial_rotation1,
        translation=initial_translation1,
        fix_rotation=False,
        fix_translation=True,
    )

    initial_spherical_cameras = world, eye0, eye1
    initial_gaze_targets = unprojected_ref_points * initial_depth

    ba = BundleAdjustment(fix_gaze_targets=False)
    success, residual, poses_in_world, gaze_targets_in_world = ba.calculate(
        initial_spherical_cameras, initial_gaze_targets
    )

    if not success:
        logger.error("Calibration solver failed to converge.")
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": solver_failed_to_converge_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    world_pose, eye0_pose, eye1_pose = poses_in_world

    sphere_pos0 = pupil0[-1]["sphere"]["center"]
    sphere_pos1 = pupil1[-1]["sphere"]["center"]
    eye0_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
    eye1_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

    observed_normals = [o.observations for o in initial_spherical_cameras]
    nearest_points = utils.calculate_nearest_points_to_targets(
        observed_normals, poses_in_world, gaze_targets_in_world
    )
    nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Binocular_Vector_Gaze_Mapper",
            "args": {
                "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
                "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
                "cal_points_3d": gaze_targets_in_world.tolist(),
                "cal_ref_points_3d": nearest_points_world.tolist(),
                "cal_gaze_points0_3d": nearest_points_eye0.tolist(),
                "cal_gaze_points1_3d": nearest_points_eye1.tolist(),
            },
        },
    )


def calibrate_3d_monocular(g_pool, matched_monocular_data, initial_depth=500):
    # monocular calibration strategy:
    # fix eye and express all points / directions in eye coordinate system
    # minimize the reprojection error by moving the world camera.

    method = "monocular 3d model"

    unprojected_ref_points, pupil_normals, _ = calibrate.preprocess_3d_data(
        matched_monocular_data, g_pool
    )

    if len(unprojected_ref_points) < 1 or len(pupil_normals) < 1:
        logger.error(not_enough_data_error_msg + " Using:" + method)
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": not_enough_data_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    unprojected_ref_points = np.asarray(unprojected_ref_points)
    pupil_normals = np.asarray(pupil_normals)

    initial_rotation_matrix, _ = utils.find_rigid_transform(
        unprojected_ref_points, pupil_normals
    )
    if matched_monocular_data[0]["pupil"]["id"] == 0:
        hardcoded_translation = eye0_hardcoded_translation
    else:
        hardcoded_translation = eye1_hardcoded_translation
    # initial_rotation and initial_translation are world cam pose in eye coordinates
    initial_rotation = cv2.Rodrigues(initial_rotation_matrix)[0].ravel()
    initial_translation = -np.dot(initial_rotation_matrix, hardcoded_translation)

    world = SphericalCamera(
        observations=unprojected_ref_points,
        rotation=initial_rotation,
        translation=initial_translation,
        fix_rotation=False,
        fix_translation=False,
    )
    eye = SphericalCamera(
        observations=pupil_normals,
        rotation=np.zeros(3),
        translation=np.zeros(3),
        fix_rotation=True,
        fix_translation=True,
    )
    initial_spherical_cameras = world, eye
    initial_gaze_targets = pupil_normals * initial_depth

    ba = BundleAdjustment(fix_gaze_targets=True)
    success, residual, poses_in_eye, gaze_targets_in_eye = ba.calculate(
        initial_spherical_cameras, initial_gaze_targets
    )

    if not success:
        logger.error("Calibration solver failed to converge.")
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": solver_failed_to_converge_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    world_pose_in_eye, eye_pose_in_eye = poses_in_eye

    # transform everything from eye coordinates to world coordinates
    # for the usage in Vector_Gaze_Mapper
    eye_pose_in_world = utils.inverse_extrinsic(world_pose_in_eye)
    poses_in_world = [np.zeros(6), eye_pose_in_world]
    gaze_targets_in_world = utils.transform_points_by_pose(
        gaze_targets_in_eye, world_pose_in_eye
    )

    sphere_pos = np.asarray(matched_monocular_data[-1]["pupil"]["sphere"]["center"])
    eye_cam_pose_in_world = utils.get_eye_cam_pose_in_world(
        eye_pose_in_world, sphere_pos
    )

    observed_normals = [o.observations for o in initial_spherical_cameras]
    nearest_points = utils.calculate_nearest_points_to_targets(
        observed_normals, poses_in_world, gaze_targets_in_world
    )
    nearest_points_world, nearest_points_eye = nearest_points

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Vector_Gaze_Mapper",
            "args": {
                "eye_camera_to_world_matrix": eye_cam_pose_in_world.tolist(),
                "cal_points_3d": gaze_targets_in_world.tolist(),
                "cal_ref_points_3d": nearest_points_world.tolist(),
                "cal_gaze_points_3d": nearest_points_eye.tolist(),
                "gaze_distance": initial_depth,
            },
        },
    )


def calibrate_2d_binocular(
    g_pool, matched_binocular_data, matched_pupil0_data, matched_pupil1_data
):
    method = "binocular polynomial regression"
    cal_pt_cloud_binocular = calibrate.preprocess_2d_data_binocular(
        matched_binocular_data
    )
    cal_pt_cloud0 = calibrate.preprocess_2d_data_monocular(matched_pupil0_data)
    cal_pt_cloud1 = calibrate.preprocess_2d_data_monocular(matched_pupil1_data)

    map_fn, inliers, params = calibrate.calibrate_2d_polynomial(
        cal_pt_cloud_binocular, g_pool.capture.frame_size, binocular=True
    )

    def create_converge_error_msg():
        return {
            "subject": "calibration.failed",
            "reason": solver_failed_to_converge_error_msg,
            "timestamp": g_pool.get_timestamp(),
            "record": True,
        }

    if not inliers.any():
        return method, create_converge_error_msg()

    map_fn, inliers, params_eye0 = calibrate.calibrate_2d_polynomial(
        cal_pt_cloud0, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, create_converge_error_msg()

    map_fn, inliers, params_eye1 = calibrate.calibrate_2d_polynomial(
        cal_pt_cloud1, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return method, create_converge_error_msg()

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Binocular_Gaze_Mapper",
            "args": {
                "params": params,
                "params_eye0": params_eye0,
                "params_eye1": params_eye1,
            },
        },
    )


def calibrate_2d_monocular(g_pool, matched_monocular_data):
    method = "monocular polynomial regression"
    cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_monocular_data)
    map_fn, inliers, params = calibrate.calibrate_2d_polynomial(
        cal_pt_cloud, g_pool.capture.frame_size, binocular=False
    )
    if not inliers.any():
        return (
            method,
            {
                "subject": "calibration.failed",
                "reason": solver_failed_to_converge_error_msg,
                "timestamp": g_pool.get_timestamp(),
                "record": True,
            },
        )

    return (
        method,
        {
            "subject": "start_plugin",
            "name": "Monocular_Gaze_Mapper",
            "args": {"params": params},
        },
    )


def match_data(g_pool, pupil_list, ref_list):
    if pupil_list and ref_list:
        pass
    else:
        logger.error(not_enough_data_error_msg)
        return {
            "subject": "calibration.failed",
            "reason": not_enough_data_error_msg,
            "timestamp": g_pool.get_timestamp(),
            "record": True,
        }

    # match eye data and check if biocular and or monocular
    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]

    # TODO unify this and don't do both
    matched_binocular_data = calibrate.closest_matches_binocular(ref_list, pupil_list)
    matched_pupil0_data = calibrate.closest_matches_monocular(ref_list, pupil0)
    matched_pupil1_data = calibrate.closest_matches_monocular(ref_list, pupil1)

    if len(matched_pupil0_data) > len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data

    logger.info(
        "Collected {} monocular calibration data.".format(len(matched_monocular_data))
    )
    logger.info(
        "Collected {} binocular calibration data.".format(len(matched_binocular_data))
    )
    return (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
        pupil0,
        pupil1,
    )


def select_calibration_method(g_pool, pupil_list, ref_list):
    len_pre_filter = len(pupil_list)
    pupil_list = [
        p for p in pupil_list if p["confidence"] >= g_pool.min_calibration_confidence
    ]
    len_post_filter = len(pupil_list)
    try:
        dismissed_percentage = 100 * (1.0 - len_post_filter / len_pre_filter)
    except ZeroDivisionError:
        pass  # empty pupil_list, is being handled in match_data
    else:
        logger.info(
            "Dismissing {:.2f}% pupil data due to confidence < {:.2f}".format(
                dismissed_percentage, g_pool.min_calibration_confidence
            )
        )

    matched_data = match_data(g_pool, pupil_list, ref_list)  # calculate matching data
    if not isinstance(matched_data, tuple):
        return None, matched_data  # matched_data is an error notification

    # unpack matching data
    (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
        pupil0,
        pupil1,
    ) = matched_data

    mode = g_pool.detection_mapping_mode

    if mode == "3d" and not (
        hasattr(g_pool.capture, "intrinsics") or g_pool.capture.intrinsics
    ):
        mode = "2d"
        logger.warning(
            "Please calibrate your world camera using 'camera intrinsics estimation' for 3d gaze mapping."
        )

    if mode == "3d":
        if matched_binocular_data:
            return calibrate_3d_binocular(
                g_pool, matched_binocular_data, pupil0, pupil1
            )
        elif matched_monocular_data:
            return calibrate_3d_monocular(g_pool, matched_monocular_data)
        else:
            logger.error(not_enough_data_error_msg)
            return (
                None,
                {
                    "subject": "calibration.failed",
                    "reason": not_enough_data_error_msg,
                    "timestamp": g_pool.get_timestamp(),
                    "record": True,
                },
            )

    elif mode == "2d":
        if matched_binocular_data:
            return calibrate_2d_binocular(
                g_pool, matched_binocular_data, matched_pupil0_data, matched_pupil1_data
            )
        elif matched_monocular_data:
            return calibrate_2d_monocular(g_pool, matched_monocular_data)
        else:
            logger.error(not_enough_data_error_msg)
            return (
                None,
                {
                    "subject": "calibration.failed",
                    "reason": not_enough_data_error_msg,
                    "timestamp": g_pool.get_timestamp(),
                    "record": True,
                },
            )


def finish_calibration(g_pool, pupil_list, ref_list):
    method, result = select_calibration_method(g_pool, pupil_list, ref_list)
    g_pool.active_calibration_plugin.notify_all(result)
    if result["subject"] != "calibration.failed":
        ts = g_pool.get_timestamp()
        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.successful",
                "method": method,
                "timestamp": ts,
                "record": True,
            }
        )

        user_calibration_data = {
            "timestamp": ts,
            "pupil_list": pupil_list,
            "ref_list": ref_list,
            "calibration_method": method,
            "mapper_name": result["name"],
            "mapper_args": result["args"],
        }

        save_object(
            user_calibration_data,
            os.path.join(g_pool.user_dir, "user_calibration_data"),
        )

        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.calibration_data",
                "record": True,
                **user_calibration_data,
            }
        )
