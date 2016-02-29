'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import numpy as np

import calibrate
from file_methods import load_object,save_object
from camera_intrinsics_estimation import load_camera_calibration

from optimization_calibration import point_line_calibration , line_line_calibration

#logging
import logging
logger = logging.getLogger(__name__)
from gaze_mappers import *


def finish_calibration(g_pool,pupil_list,ref_list):
    not_enough_data_error_msg = 'Did not collect enough data during calibration.'

    if pupil_list and ref_list:
        pass
    else:
        logger.error(not_enough_data_error_msg)
        g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
        return

    camera_intrinsics = load_camera_calibration(g_pool)

    # match eye data and check if biocular and or monocular
    pupil0 = [p for p in pupil_list if p['id']==0]
    pupil1 = [p for p in pupil_list if p['id']==1]

    #TODO unify this and don't do both
    matched_binocular_data = calibrate.closest_matches_binocular(ref_list,pupil_list)
    matched_pupil0_data = calibrate.closest_matches_monocular(ref_list,pupil0)
    matched_pupil1_data = calibrate.closest_matches_monocular(ref_list,pupil1)

    if len(matched_pupil0_data)>len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data

    logger.info('Collected %s monocular calibration data.'%len(matched_monocular_data))
    logger.info('Collected %s binocular calibration data.'%len(matched_binocular_data))


    mode = g_pool.detection_mapping_mode

    if mode == '3d' and not camera_intrinsics:
        mode = '2d'
        logger.warning("Please calibrate your world camera using 'camera intrinsics estimation' for 3d gaze mapping.")

    if mode == '3d':
        if matched_binocular_data:
            method = 'binocular 3d model'
            ref_3d, gaze0_3d, gaze1_3d = calibrate.preprocess_3d_data(matched_binocular_data,
                                            camera_intrinsics = camera_intrinsics )

            if len(ref_3d) < 1 or len(gaze0_3d) < 1 or len(gaze1_3d) < 1:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            sphere_pos0 = pupil0[-1]['sphere']['center']
            sphere_pos1 = pupil1[-1]['sphere']['center']
            initial_orientation0 = [ 0.05334223 , 0.93651217 , 0.07765971 ,-0.33774033] #eye0
            initial_orientation1 = [ 0.34200577 , 0.21628107 , 0.91189657 ,   0.06855066] #eye1
            initial_translation0 = (25, 15, -10)
            initial_translation1 = (-5, 15, -10)

            #this returns the translation of the eye and not of the camera coordinate system
            #need to take sphere position into account
            #success, orientation, translation , avg_distance  = line_line_calibration(ref_3d,  gaze_3d, initial_orientation, initial_translation)
            success0, orientation0, translation0 , _  = line_line_calibration(ref_3d,  gaze0_3d, initial_orientation0, initial_translation0)
            success1, orientation1, translation1 , _  = line_line_calibration(ref_3d,  gaze1_3d, initial_orientation1, initial_translation1)
            orientation0 = np.array(orientation0)
            translation0 = np.array(translation0)
            orientation1 = np.array(orientation1)
            translation1 = np.array(translation1)

            # print 'orientation: ' , orientation
            # print 'translation: ' , translation
            # print 'avg distance: ' , avg_distance

            if not success0 and not success1:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            rotation_matrix0 = calibrate.quat2mat(orientation0)
            rotation_matrix1 = calibrate.quat2mat(orientation1)
            # we need to take the sphere position into account
            # orientation and translation are referring to the sphere center.
            # but we wanna have it referring to the camera center

            # since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates
            sphere_translation0 = np.array( sphere_pos0 )
            sphere_translation_world0 = np.dot( rotation_matrix0 , sphere_translation0)
            camera_translation0 = translation0 - sphere_translation_world0
            eye_camera_to_world_matrix0  = np.matrix(np.eye(4))
            eye_camera_to_world_matrix0[:3,:3] = rotation_matrix0
            eye_camera_to_world_matrix0[:3,3:4] = np.reshape(camera_translation0, (3,1) )
            world_to_eye_matrix0  = np.matrix(np.eye(4))
            world_to_eye_matrix0[:3,:3] = rotation_matrix0.T
            world_to_eye_matrix0[:3,3:4] = np.reshape(-np.dot(rotation_matrix0.T , camera_translation0), (3,1) )

            sphere_translation1 = np.array( sphere_pos1 )
            sphere_translation_world1 = np.dot( rotation_matrix1 , sphere_translation1)
            camera_translation1 = translation1 - sphere_translation_world1
            eye_camera_to_world_matrix1  = np.matrix(np.eye(4))
            eye_camera_to_world_matrix1[:3,:3] = rotation_matrix1
            eye_camera_to_world_matrix1[:3,3:4] = np.reshape(camera_translation1, (3,1) )
            world_to_eye_matrix1  = np.matrix(np.eye(4))
            world_to_eye_matrix1[:3,:3] = rotation_matrix1.T
            world_to_eye_matrix1[:3,3:4] = np.reshape(-np.dot(rotation_matrix1.T , camera_translation1), (3,1) )


            gaze0_points_3d = []
            gaze1_points_3d = []
            ref_points_3d = []
            avg_error = 0.0
            avg_gaze_distance = 0.0
            avg_ref_distance = 0.0
            for i in range(0,len(ref_3d)):
                ref_v = ref_3d[i]
                gaze_direction0 = gaze0_3d[i]
                gaze_direction_world0 = np.dot(rotation_matrix0, gaze_direction0)
                gaze_line0 = ( translation0 , translation0 + gaze_direction_world0 )

                gaze_direction1 = gaze1_3d[i]
                gaze_direction_world1 = np.dot(rotation_matrix1, gaze_direction1)
                gaze_line1 = ( translation1 , translation1 + gaze_direction_world1 )

                ref_line = ( np.zeros(3) , ref_v )

                gaze_intersection, distance = calibrate.nearest_intersection( gaze_line0 , gaze_line1 )
                ref_point, distance_ref  = calibrate.nearest_linepoint_to_point( gaze_intersection, ref_line )

                gaze0_point_world, distance0 = calibrate.nearest_linepoint_to_point( ref_point, gaze_line0 )
                gaze1_point_world, distance1 = calibrate.nearest_linepoint_to_point( ref_point, gaze_line1 )

                gaze0_points_3d.append( gaze0_point_world )
                gaze1_points_3d.append( gaze1_point_world )
                ref_points_3d.append( ref_point )

                avg_error += (distance_ref + distance0 + distance1)/ 3.0
                avg_ref_distance += np.linalg.norm(ref_point)
                avg_gaze_distance += (np.linalg.norm(gaze0_point_world) + np.linalg.norm(gaze1_point_world) ) * 0.5


            avg_error /= len(ref_3d)
            avg_gaze_distance /= len(ref_3d)
            avg_ref_distance /= len(ref_3d)
            avg_distance = (avg_ref_distance + avg_gaze_distance) * 0.5
            logger.info('calibration average error: %s'%avg_error)
            logger.info('calibration average distance: %s'%avg_distance)

            print 'avg error: ' , avg_error
            print 'avg gaze distance: ' , avg_gaze_distance
            print 'avg ref distance: ' , avg_ref_distance

            g_pool.plugins.add(Binocular_Vector_Gaze_Mapper,args={'eye_camera_to_world_matrix0':eye_camera_to_world_matrix0,'eye_camera_to_world_matrix1':eye_camera_to_world_matrix1 , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_3d, 'cal_gaze_points0_3d': gaze0_points_3d, 'cal_gaze_points1_3d': gaze1_points_3d })


        elif matched_monocular_data:
            method = 'monocular 3d model'
            ref_3d , gaze_3d, _ = calibrate.preprocess_3d_data(matched_monocular_data,
                                            camera_intrinsics = camera_intrinsics )

            if len(ref_3d) < 1 or len(gaze_3d) < 1:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            if  matched_monocular_data[-1]['pupil']['id'] == 0:
                initial_orientation = [ 0.05334223 , 0.93651217 , 0.07765971 ,-0.33774033] #eye0
                initial_translation = (10, 30, -10)
            else:
                initial_orientation = [ 0.34200577 , 0.21628107 , 0.91189657 ,   0.06855066] #eye1
                initial_translation = (-50, 30, -10)


            #this returns the translation of the eye and not of the camera coordinate system
            #need to take sphere position into account
            #success, orientation, translation , avg_distance  = line_line_calibration(ref_3d,  gaze_3d, initial_orientation, initial_translation)
            success, orientation, translation , _  = line_line_calibration(ref_3d,  gaze_3d, initial_orientation, initial_translation)
            orientation = np.array(orientation)
            translation = np.array(translation)
            # print 'orientation: ' , orientation
            # print 'translation: ' , translation
            # print 'avg distance: ' , avg_distance

            if not success:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            rotation_matrix = calibrate.quat2mat(orientation)
            # we need to take the sphere position into account
            # orientation and translation are referring to the sphere center.
            # but we wanna have it referring to the camera center

            # since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates
            sphere_translation = np.array( matched_monocular_data[-1]['pupil']['sphere']['center'] )
            sphere_translation_world = np.dot( rotation_matrix , sphere_translation)
            camera_translation = translation - sphere_translation_world
            eye_camera_to_world_matrix  = np.matrix(np.eye(4))
            eye_camera_to_world_matrix[:3,:3] = rotation_matrix
            eye_camera_to_world_matrix[:3,3:4] = np.reshape(camera_translation, (3,1) )
            world_to_eye_matrix  = np.matrix(np.eye(4))
            world_to_eye_matrix[:3,:3] = rotation_matrix.T
            world_to_eye_matrix[:3,3:4] = np.reshape(-np.dot(rotation_matrix.T , camera_translation), (3,1) )

            # print eye_camera_to_world_matrix
            # print world_to_eye_matrix
            gaze_points_3d = []
            ref_points_3d = []
            avg_error = 0.0
            avg_gaze_distance = 0.0
            avg_ref_distance = 0.0
            for i in range(0,len(ref_3d)):
                ref_p = ref_3d[i]
                gaze_direction = gaze_3d[i]
                gaze_direction_world = np.dot(rotation_matrix, gaze_direction)
                gaze_line = ( translation , translation + gaze_direction_world )
                ref_line = ( np.zeros(3) , ref_p )
                ref_point_world , gaze_point_world , distance = calibrate.nearest_intersection_points( ref_line , gaze_line )

                gaze_points_3d.append( gaze_point_world )
                ref_points_3d.append( ref_point_world )

                avg_error += distance
                avg_ref_distance += np.linalg.norm(ref_point_world)
                avg_gaze_distance += np.linalg.norm(gaze_point_world)


            avg_error /= len(ref_3d)
            avg_gaze_distance /= len(ref_3d)
            avg_ref_distance /= len(ref_3d)
            avg_distance = (avg_ref_distance + avg_gaze_distance) * 0.5
            logger.info('calibration average error: %s'%avg_error)
            logger.info('calibration average distance: %s'%avg_distance)

            # print 'avg error: ' , avg_error
            # print 'avg gaze distance: ' , avg_gaze_distance
            # print 'avg ref distance: ' , avg_ref_distance

            g_pool.plugins.add(Vector_Gaze_Mapper,args=
                {'eye_camera_to_world_matrix':eye_camera_to_world_matrix , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_points_3d,
                 'cal_gaze_points_3d': gaze_points_3d})

        else:
            logger.error(not_enough_data_error_msg)
            g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
            return

    elif mode == '2d':
        if matched_binocular_data:
            method = 'binocular polynomial regression'
            cal_pt_cloud_binocular = calibrate.preprocess_2d_data_binocular(matched_binocular_data)
            cal_pt_cloud0 = calibrate.preprocess_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = calibrate.preprocess_2d_data_monocular(matched_pupil1_data)
            map_fn,inliers,params = calibrate.calibrate_2d_polynomial(cal_pt_cloud_binocular,g_pool.capture.frame_size,binocular=True)
            map_fn,inliers,params_eye0 = calibrate.calibrate_2d_polynomial(cal_pt_cloud0,g_pool.capture.frame_size,binocular=False)
            map_fn,inliers,params_eye1 = calibrate.calibrate_2d_polynomial(cal_pt_cloud1,g_pool.capture.frame_size,binocular=False)
            g_pool.plugins.add(Binocular_Gaze_Mapper,args={'params':params, 'params_eye0':params_eye0, 'params_eye1':params_eye1})


        elif matched_monocular_data:
            method = 'monocular polynomial regression'
            cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_monocular_data)
            map_fn,inliers,params = calibrate.calibrate_2d_polynomial(cal_pt_cloud,g_pool.capture.frame_size,binocular=False)
            g_pool.plugins.add(Simple_Gaze_Mapper,args={'params':params})
        else:
            logger.error(not_enough_data_error_msg)
            g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
            return

    user_calibration_data = {'pupil_list':pupil_list,'ref_list':ref_list,'calibration_method':method}
    save_object(user_calibration_data,os.path.join(g_pool.user_dir, "user_calibration_data"))
    g_pool.active_calibration_plugin.notify_all({'subject':'calibration_successful','method':method,'timestamp':g_pool.capture.get_timestamp(),'record':True})

