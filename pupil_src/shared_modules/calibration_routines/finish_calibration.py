'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import numpy as np

from . import calibrate
from math_helper import *
from file_methods import load_object,save_object
from . camera_intrinsics_estimation import load_camera_calibration

from . optimization_calibration import bundle_adjust_calibration
from . calibrate import find_rigid_transform
#logging
import logging
logger = logging.getLogger(__name__)
from . gaze_mappers import *

not_enough_data_error_msg = 'Did not collect enough data during calibration.'
solver_failed_to_converge_error_msg = 'Paramters could not be estimated from data.'


def calibrate_3d_binocular(g_pool, camera_intrinsics, matched_binocular_data, pupil0, pupil1):
    method = 'binocular 3d model'
    hardcoded_translation0 = np.array([20, 15, -20])
    hardcoded_translation1 = np.array([-40, 15, -20])

    # TODO model the world as cv2 pinhole camera with distorion and focal in ceres.
    # right now we solve using a few permutations of K
    smallest_residual = 1000
    scales = list(np.linspace(0.7, 1.4, 20))
    K = np.asarray(camera_intrinsics["camera_matrix"])

    for s in scales:
        scale = np.ones(K.shape)
        scale[0, 0] *= s
        scale[1, 1] *= s
        camera_intrinsics["camera_matrix"] = K*scale

        ref_dir, gaze0_dir, gaze1_dir = calibrate.preprocess_3d_data(matched_binocular_data,
                                                                     camera_intrinsics=camera_intrinsics)

        if len(ref_dir) < 1 or len(gaze0_dir) < 1 or len(gaze1_dir) < 1:
            logger.error(not_enough_data_error_msg)
            return method, {'subject': 'calibration.failed', 'reason': not_enough_data_error_msg,
                            'timestamp': g_pool.get_timestamp(), 'record': True}

        sphere_pos0 = pupil0[-1]['sphere']['center']
        sphere_pos1 = pupil1[-1]['sphere']['center']

        initial_R0, initial_t0 = find_rigid_transform(np.array(gaze0_dir)*500,np.array(ref_dir)*500)
        initial_rotation0 = math_helper.quaternion_from_rotation_matrix(initial_R0)
        # initial_translation0 = np.array(initial_t0).reshape(3)  # currently not used

        initial_R1, initial_t1 = find_rigid_transform(np.array(gaze1_dir)*500,np.array(ref_dir)*500)
        initial_rotation1 = math_helper.quaternion_from_rotation_matrix(initial_R1)
        # initial_translation1 = np.array(initial_t1).reshape(3)  # currently not used

        eye0 = {"observations": gaze0_dir, "translation": hardcoded_translation0,
                "rotation": initial_rotation0, 'fix': ['translation']}

        eye1 = {"observations": gaze1_dir, "translation": hardcoded_translation1,
                "rotation": initial_rotation1, 'fix': ['translation']}
        world = {"observations": ref_dir, "translation": (0, 0, 0),
                 "rotation": (1, 0, 0, 0), 'fix': ['translation', 'rotation']}
        initial_observers = [eye0, eye1, world]
        initial_points = np.array(ref_dir)*500

        success, residual, observers, points = bundle_adjust_calibration(initial_observers,
                                                                         initial_points,
                                                                         fix_points=False)

        if residual <= smallest_residual:
            smallest_residual = residual
            scales[-1] = s

    if not success:
        logger.error("Calibration solver faild to converge.")
        return method, {'subject': 'calibration.failed', 'reason': solver_failed_to_converge_error_msg,
                        'timestamp': g_pool.get_timestamp(), 'record': True}

    eye0, eye1, world = observers

    t_world0 = np.array(eye0['translation'])
    R_world0 = math_helper.quaternion_rotation_matrix(np.array(eye0['rotation']))
    t_world1 = np.array(eye1['translation'])
    R_world1 = math_helper.quaternion_rotation_matrix(np.array(eye1['rotation']))

    def toWorld0(p):
        return np.dot(R_world0, p)+t_world0

    def toWorld1(p):
        return np.dot(R_world1, p)+t_world1

    points_a = []  # world coords
    points_b = []  # eye0 coords
    points_c = []  # eye1 coords
    for a, b, c, point in zip(world['observations'], eye0['observations'],
                              eye1['observations'], points):
        line_a = np.array([0, 0, 0]), np.array(a)  # observation as line
        line_b = toWorld0(np.array([0, 0, 0])), toWorld0(b)  # eye0 observation line in world coords
        line_c = toWorld1(np.array([0, 0, 0])), toWorld1(c)  # eye1 observation line in world coords
        close_point_a, _ = math_helper.nearest_linepoint_to_point(point, line_a)
        close_point_b, _ = math_helper.nearest_linepoint_to_point(point, line_b)
        close_point_c, _ = math_helper.nearest_linepoint_to_point(point, line_c)
        points_a.append(close_point_a.tolist())
        points_b.append(close_point_b.tolist())
        points_c.append(close_point_c.tolist())

    # we need to take the sphere position into account
    # orientation and translation are referring to the sphere center.
    # but we want to have it referring to the camera center
    # since the actual translation is in world coordinates, the sphere
    # translation needs to be calculated in world coordinates
    sphere_translation = np.array(sphere_pos0)
    sphere_translation_world = np.dot(R_world0, sphere_translation)
    camera_translation = t_world0 - sphere_translation_world
    eye_camera_to_world_matrix0 = np.eye(4)
    eye_camera_to_world_matrix0[:3, :3] = R_world0
    eye_camera_to_world_matrix0[:3, 3:4] = np.reshape(camera_translation, (3, 1))

    sphere_translation = np.array(sphere_pos1)
    sphere_translation_world = np.dot(R_world1, sphere_translation)
    camera_translation = t_world1 - sphere_translation_world
    eye_camera_to_world_matrix1 = np.eye(4)
    eye_camera_to_world_matrix1[:3, :3] = R_world1
    eye_camera_to_world_matrix1[:3, 3:4] = np.reshape(camera_translation, (3, 1))

    camera_intrinsics['camera_matrix'] = camera_intrinsics['camera_matrix'].tolist()
    return method, {'subject': 'start_plugin', 'name': 'Binocular_Vector_Gaze_Mapper',
                    'args': {'eye_camera_to_world_matrix0': eye_camera_to_world_matrix0.tolist(),
                             'eye_camera_to_world_matrix1': eye_camera_to_world_matrix1.tolist(),
                             'camera_intrinsics': camera_intrinsics,
                             'cal_points_3d': points,
                             'cal_ref_points_3d': points_a,
                             'cal_gaze_points0_3d': points_b,
                             'cal_gaze_points1_3d': points_c}}


def calibrate_3d_monocular(g_pool, camera_intrinsics, matched_monocular_data):
    method = 'monocular 3d model'
    hardcoded_translation0 = np.array([20, 15, -20])
    hardcoded_translation1 = np.array([-40, 15, -20])
    # TODO model the world as cv2 pinhole camera with distorion and focal in ceres.
    # right now we solve using a few permutations of K
    smallest_residual = 1000
    scales = list(np.linspace(0.7, 1.4, 20))
    K = np.asarray(camera_intrinsics["camera_matrix"])
    for s in scales:
        scale = np.ones(K.shape)
        scale[0, 0] *= s
        scale[1, 1] *= s
        camera_intrinsics["camera_matrix"] = K*scale
        ref_dir, gaze_dir, _ = calibrate.preprocess_3d_data(matched_monocular_data,
                                                            camera_intrinsics=camera_intrinsics)
        # save_object((ref_dir,gaze_dir),os.path.join(g_pool.user_dir, "testdata"))
        if len(ref_dir) < 1 or len(gaze_dir) < 1:
            logger.error(not_enough_data_error_msg + " Using:" + method)
            return method, {'subject': 'calibration.failed', 'reason': not_enough_data_error_msg,
                            'timestamp': g_pool.get_timestamp(), 'record': True}

        # monocular calibration strategy: mimize the reprojection error by moving the world camera.
        # we fix the eye points and work in the eye coord system.
        initial_R, initial_t = find_rigid_transform(np.array(ref_dir)*500, np.array(gaze_dir)*500)
        initial_rotation = math_helper.quaternion_from_rotation_matrix(initial_R)
        # initial_translation = np.array(initial_t).reshape(3)  # currently not used
        # this problem is scale invariant so we scale to some sensical value.

        if matched_monocular_data[0]['pupil']['id'] == 0:
            hardcoded_translation = hardcoded_translation0
        else:
            hardcoded_translation = hardcoded_translation1

        eye = {"observations": gaze_dir,
               "translation": (0, 0, 0),
               "rotation": (1, 0, 0, 0),
               'fix': ['translation', 'rotation']}

        world = {"observations": ref_dir,
                 "translation": np.dot(initial_R, -hardcoded_translation),
                 "rotation": initial_rotation,
                 'fix': ['translation']}

        initial_observers = [eye, world]
        initial_points = np.array(gaze_dir)*500

        success, residual, observers, points_in_eye = bundle_adjust_calibration(initial_observers,
                                                                                initial_points,
                                                                                fix_points=True)
        if residual <= smallest_residual:
            smallest_residual = residual
            scales[-1] = s

    eye, world = observers

    if not success:
        logger.error("Calibration solver faild to converge.")
        return method, {'subject': 'calibration.failed', 'reason': solver_failed_to_converge_error_msg,
                        'timestamp': g_pool.get_timestamp(), 'record': True}

    # pose of the world in eye coords.
    rotation = np.array(world['rotation'])
    t_world = np.array(world['translation'])
    R_world = math_helper.quaternion_rotation_matrix(rotation)

    # inverse is pose of eye in world coords
    R_eye = R_world.T
    t_eye = np.dot(R_eye, -t_world)

    def toWorld(p):
        return np.dot(R_eye, p)+np.array(t_eye)

    points_in_world = [toWorld(p).tolist() for p in points_in_eye]

    points_a = []  # world coords
    points_b = []  # cam2 coords
    for a, b, point in zip(world['observations'], eye['observations'], points_in_world):

        line_a = np.array([0, 0, 0]), np.array(a)  # observation as line
        line_b = toWorld(np.array([0, 0, 0])), toWorld(b)  # cam2 observation line in cam1 coords
        close_point_a, _ = math_helper.nearest_linepoint_to_point(point, line_a)
        close_point_b, _ = math_helper.nearest_linepoint_to_point(point, line_b)
        # print np.linalg.norm(point-close_point_a),np.linalg.norm(point-close_point_b)

        points_a.append(close_point_a.tolist())
        points_b.append(close_point_b.tolist())

    # we need to take the sphere position into account
    # orientation and translation are referring to the sphere center.
    # but we want to have it referring to the camera center
    # since the actual translation is in world coordinates, the sphere
    # translation needs to be calculated in world coordinates
    sphere_translation = np.array(matched_monocular_data[-1]['pupil']['sphere']['center'])
    sphere_translation_world = np.dot(R_eye, sphere_translation)
    camera_translation = t_eye - sphere_translation_world
    eye_camera_to_world_matrix = np.eye(4)
    eye_camera_to_world_matrix[:3, :3] = R_eye
    eye_camera_to_world_matrix[:3, 3:4] = np.reshape(camera_translation, (3, 1))

    camera_intrinsics['camera_matrix'] = camera_intrinsics['camera_matrix'].tolist()
    return method, {'subject': 'start_plugin', 'name': 'Vector_Gaze_Mapper',
                    'args': {'eye_camera_to_world_matrix': eye_camera_to_world_matrix.tolist(),
                             'camera_intrinsics': camera_intrinsics,
                             'cal_points_3d': points_in_world,
                             'cal_ref_points_3d': points_a,
                             'cal_gaze_points_3d': points_b,
                             'gaze_distance': 500}}


def calibrate_2d_binocular(g_pool, matched_binocular_data, matched_pupil0_data, matched_pupil1_data):
    method = 'binocular polynomial regression'
    cal_pt_cloud_binocular = calibrate.preprocess_2d_data_binocular(matched_binocular_data)
    cal_pt_cloud0 = calibrate.preprocess_2d_data_monocular(matched_pupil0_data)
    cal_pt_cloud1 = calibrate.preprocess_2d_data_monocular(matched_pupil1_data)

    map_fn, inliers, params = calibrate.calibrate_2d_polynomial(cal_pt_cloud_binocular,
                                                                g_pool.capture.frame_size,
                                                                binocular=True)

    def create_converge_error_msg():
        return {'subject': 'calibration.failed', 'reason': solver_failed_to_converge_error_msg,
                'timestamp': g_pool.get_timestamp(), 'record': True}

    if not inliers.any():
        return method, create_converge_error_msg()

    map_fn, inliers, params_eye0 = calibrate.calibrate_2d_polynomial(cal_pt_cloud0,
                                                                     g_pool.capture.frame_size,
                                                                     binocular=False)
    if not inliers.any():
        return method, create_converge_error_msg()

    map_fn, inliers, params_eye1 = calibrate.calibrate_2d_polynomial(cal_pt_cloud1,
                                                                     g_pool.capture.frame_size,
                                                                     binocular=False)
    if not inliers.any():
        return method, create_converge_error_msg()

    return method, {'subject': 'start_plugin', 'name': 'Binocular_Gaze_Mapper',
                    'args': {'params': params, 'params_eye0': params_eye0, 'params_eye1': params_eye1}}


def calibrate_2d_monocular(g_pool, matched_monocular_data):
    method = 'monocular polynomial regression'
    cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_monocular_data)
    map_fn, inliers, params = calibrate.calibrate_2d_polynomial(cal_pt_cloud,
                                                                g_pool.capture.frame_size,
                                                                binocular=False)
    if not inliers.any():
        return method, {'subject': 'calibration.failed', 'reason': solver_failed_to_converge_error_msg,
                        'timestamp': g_pool.get_timestamp(), 'record': True}

    return method, {'subject': 'start_plugin', 'name': 'Monocular_Gaze_Mapper', 'args': {'params': params}}


def match_data(g_pool, pupil_list, ref_list):
    if pupil_list and ref_list:
        pass
    else:
        logger.error(not_enough_data_error_msg)
        return {'subject': 'calibration.failed', 'reason': not_enough_data_error_msg,
                'timestamp': g_pool.get_timestamp(), 'record': True}

    camera_intrinsics = load_camera_calibration(g_pool)

    # match eye data and check if biocular and or monocular
    pupil0 = [p for p in pupil_list if p['id'] == 0]
    pupil1 = [p for p in pupil_list if p['id'] == 1]

    # TODO unify this and don't do both
    matched_binocular_data = calibrate.closest_matches_binocular(ref_list, pupil_list)
    matched_pupil0_data = calibrate.closest_matches_monocular(ref_list, pupil0)
    matched_pupil1_data = calibrate.closest_matches_monocular(ref_list, pupil1)

    if len(matched_pupil0_data) > len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data

    logger.info('Collected {} monocular calibration data.'.format(len(matched_monocular_data)))
    logger.info('Collected {} binocular calibration data.'.format(len(matched_binocular_data)))
    return (camera_intrinsics, matched_binocular_data, matched_monocular_data,
            matched_pupil0_data, matched_pupil1_data, pupil0, pupil1)


def select_calibration_method(g_pool, pupil_list, ref_list):
    matched_data = match_data(g_pool, pupil_list, ref_list)  # calculate matching data
    if not isinstance(matched_data, tuple):
        return None, matched_data  # matched_data is a error notification

    # unpack matching data
    (camera_intrinsics, matched_binocular_data, matched_monocular_data,
        matched_pupil0_data, matched_pupil1_data, pupil0, pupil1) = matched_data

    mode = g_pool.detection_mapping_mode

    if mode == '3d' and not camera_intrinsics:
        mode = '2d'
        logger.warning("Please calibrate your world camera using 'camera intrinsics estimation' for 3d gaze mapping.")

    if mode == '3d':
        if matched_binocular_data:
            return calibrate_3d_binocular(g_pool, camera_intrinsics, matched_binocular_data, pupil0, pupil1)
        elif matched_monocular_data:
            return calibrate_3d_monocular(g_pool, camera_intrinsics, matched_monocular_data)
        else:
            logger.error(not_enough_data_error_msg)
            return None, {'subject': 'calibration.failed', 'reason': not_enough_data_error_msg,
                          'timestamp': g_pool.get_timestamp(), 'record': True}

    elif mode == '2d':
        if matched_binocular_data:
            return calibrate_2d_binocular(g_pool, matched_binocular_data, matched_pupil0_data, matched_pupil1_data)
        elif matched_monocular_data:
            return calibrate_2d_monocular(g_pool, matched_monocular_data)
        else:
            logger.error(not_enough_data_error_msg)
            return None, {'subject': 'calibration.failed', 'reason': not_enough_data_error_msg,
                          'timestamp': g_pool.get_timestamp(), 'record': True}


def finish_calibration(g_pool, pupil_list, ref_list):
    method, result = select_calibration_method(g_pool, pupil_list, ref_list)
    g_pool.active_calibration_plugin.notify_all(result)
    if result['subject'] != 'calibration.failed':
        ts = g_pool.get_timestamp()
        g_pool.active_calibration_plugin.notify_all({'subject': 'calibration.successful',
                                                     'method': method,
                                                     'timestamp': ts,
                                                     'record': True})

        g_pool.active_calibration_plugin.notify_all({'subject': 'calibration.calibration_data',
                                                     'timestamp': ts,
                                                     'pupil_list': pupil_list,
                                                     'ref_list': ref_list,
                                                     'calibration_method': method,
                                                     'record': True})

        # this is only used by show calibration. TODO: rewrite show calibraiton.
        user_calibration_data = {'timestamp': ts, 'pupil_list': pupil_list,
                                 'ref_list': ref_list, 'calibration_method': method}

        save_object(user_calibration_data, os.path.join(g_pool.user_dir, "user_calibration_data"))
