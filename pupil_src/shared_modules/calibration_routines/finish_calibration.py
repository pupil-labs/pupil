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
import math_helper
from file_methods import load_object,save_object
from camera_intrinsics_estimation import load_camera_calibration

from optimization_calibration import point_line_calibration  , bundle_adjust_calibration
from calibrate import find_rigid_transform
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
            ref_dir, gaze0_dir, gaze1_dir = calibrate.preprocess_3d_data(matched_binocular_data,
                                            camera_intrinsics = camera_intrinsics )

            if len(ref_dir) < 1 or len(gaze0_dir) < 1 or len(gaze1_dir) < 1:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            sphere_pos0 = pupil0[-1]['sphere']['center']
            sphere_pos1 = pupil1[-1]['sphere']['center']

            initial_R0,initial_t0 = find_rigid_transform(np.array(gaze0_dir),np.array(ref_dir))
            initial_orientation0 = math_helper.quaternion_from_rotation_matrix(initial_R0)
            initial_translation0 = np.array(initial_t0).reshape(3)
            # this problem is scale invariant so we scale to some sensical value.
            initial_translation0 *= 30/np.linalg.norm(initial_translation0)

            initial_R1,initial_t1 = find_rigid_transform(np.array(gaze1_dir),np.array(ref_dir))
            initial_orientation1 = math_helper.quaternion_from_rotation_matrix(initial_R1)
            initial_translation1 = np.array(initial_t1).reshape(3)
            # this problem is scale invariant so we scale to some sensical value.
            initial_translation1 *= 50/np.linalg.norm(initial_translation1)


            #this returns the translation of the eye and not of the camera coordinate system
            #need to take sphere position into account
            success0, orientation0, translation0 , solver_residual0  = line_line_calibration(ref_dir,  gaze0_dir, initial_orientation0, initial_translation0 , use_weight = True)
            success1, orientation1, translation1 , solver_residual1  = line_line_calibration(ref_dir,  gaze1_dir, initial_orientation1, initial_translation1,  use_weight = True)

            # overwrite solution with intial guess
            # success0, orientation0, translation0 , avg_distance0 = True, initial_orientation0,initial_translation0,-1
            # success1, orientation1, translation1 , avg_distance1 = True, initial_orientation1,initial_translation1,-1



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

            rotation_matrix0 = math_helper.quaternion_rotation_matrix(orientation0)
            rotation_matrix1 = math_helper.quaternion_rotation_matrix(orientation1)
            # we need to take the sphere position into account
            # orientation and translation are referring to the sphere center.
            # but we want to have it referring to the camera center

            # since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates
            sphere_translation0 = np.array( sphere_pos0 )
            sphere_translation_world0 = np.dot( rotation_matrix0 , sphere_translation0)
            camera_translation0 = translation0 - sphere_translation_world0
            eye_camera_to_world_matrix0  = np.eye(4)
            eye_camera_to_world_matrix0[:3,:3] = rotation_matrix0
            eye_camera_to_world_matrix0[:3,3:4] = np.reshape(camera_translation0, (3,1) )

            sphere_translation1 = np.array( sphere_pos1 )
            sphere_translation_world1 = np.dot( rotation_matrix1 , sphere_translation1)
            camera_translation1 = translation1 - sphere_translation_world1
            eye_camera_to_world_matrix1  = np.eye(4)
            eye_camera_to_world_matrix1[:3,:3] = rotation_matrix1
            eye_camera_to_world_matrix1[:3,3:4] = np.reshape(camera_translation1, (3,1) )


            gaze0_points_3d = []
            gaze1_points_3d = []
            ref_points_3d = []
            avg_error = 0.0
            avg_gaze_distance = 0.0
            avg_ref_distance = 0.0
            for i in range(0,len(ref_dir)):
                ref_v = ref_dir[i]
                gaze_direction0 = gaze0_dir[i]
                gaze_direction_world0 = np.dot(rotation_matrix0, gaze_direction0)
                gaze_line0 = ( translation0 , translation0 + gaze_direction_world0 )

                gaze_direction1 = gaze1_dir[i]
                gaze_direction_world1 = np.dot(rotation_matrix1, gaze_direction1)
                gaze_line1 = ( translation1 , translation1 + gaze_direction_world1 )

                ref_line = ( np.zeros(3) , ref_v )

                r0, _ , _ = math_helper.nearest_intersection_points(ref_line , gaze_line0)
                r1, _ , _ = math_helper.nearest_intersection_points(ref_line , gaze_line1)
                ref_point = (r0+r1)*0.5

                gaze0_point_world, distance0 = math_helper.nearest_intersection( ref_line, gaze_line0 )
                gaze1_point_world, distance1 = math_helper.nearest_intersection( ref_line, gaze_line1 )

                gaze0_points_3d.append( gaze0_point_world )
                gaze1_points_3d.append( gaze1_point_world )
                ref_points_3d.append( ref_point )

                avg_error += ( distance0 + distance1)/ 2.0
                avg_ref_distance += np.linalg.norm(ref_point)
                avg_gaze_distance += (np.linalg.norm(gaze0_point_world) + np.linalg.norm(gaze1_point_world) ) * 0.5


            avg_error /= len(ref_dir)
            avg_gaze_distance /= len(ref_dir)
            avg_ref_distance /= len(ref_dir)
            avg_distance = (avg_ref_distance + avg_gaze_distance) * 0.5
            logger.info('calibration average error: %s'%avg_error)
            logger.info('calibration average distance: %s'%avg_distance)

            print 'avg error: ' , avg_error
            print 'avg gaze distance: ' , avg_gaze_distance
            print 'avg ref distance: ' , avg_ref_distance

            g_pool.plugins.add(Binocular_Vector_Gaze_Mapper,args={'eye_camera_to_world_matrix0':eye_camera_to_world_matrix0,'eye_camera_to_world_matrix1':eye_camera_to_world_matrix1 , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_points_3d, 'cal_gaze_points0_3d': gaze0_points_3d, 'cal_gaze_points1_3d': gaze1_points_3d ,'manual_gaze_distance':avg_distance})


        elif matched_monocular_data:
            method = 'monocular 3d model'
            ref_dir , gaze_dir, _ = calibrate.preprocess_3d_data(matched_monocular_data,
                                            camera_intrinsics = camera_intrinsics )

            if len(ref_dir) < 1 or len(gaze_dir) < 1:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            initial_R,initial_t = find_rigid_transform(np.array(gaze_dir),np.array(ref_dir))
            initial_orientation = math_helper.quaternion_from_rotation_matrix(initial_R)
            initial_translation = np.array(initial_t).reshape(3)
            # this problem is scale invariant so we scale to some sensical value.
            initial_translation *= 30/np.linalg.norm(initial_translation)


            o1 = { "directions" : ref_dir , "translation" : (0,0,0) , "orientation" : (1,0,0,0)  }
            o2 = { "directions" : gaze_dir , "translation" : initial_translation , "orientation" : initial_orientation  }
            observations = [o1, o2]

            #this returns the translation of the eye and not of the camera coordinate system
            #need to take sphere position into account
            success, orientations, translations , avg_distance  = bundle_adjust_calibration(observations , use_weight = True )
            print 'solver_residual: ',avg_distance
            # overwrite solution with intial guess
            # success, orientation, translation , avg_distance = True, initial_orientation,initial_translation,-1


            orientation = np.array(orientations[1]) #first one is fixed world camera
            translation = np.array(translations[1]) #first one is fixed world camera
            # print 'orientation: ' , orientation
            # print 'translation: ' , translation
            # print 'avg distance: ' , avg_distance

            if not success:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            rotation_matrix = math_helper.quaternion_rotation_matrix(orientation)


            # we need to take the sphere position into account
            # orientation and translation are referring to the sphere center.
            # but we want to have it referring to the camera center

            # since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates
            sphere_translation = np.array( matched_monocular_data[-1]['pupil']['sphere']['center'] )
            sphere_translation_world = np.dot( rotation_matrix , sphere_translation)
            camera_translation = translation - sphere_translation_world
            eye_camera_to_world_matrix  = np.eye(4)
            eye_camera_to_world_matrix[:3,:3] = rotation_matrix
            eye_camera_to_world_matrix[:3,3:4] = np.reshape(camera_translation, (3,1) )


            gaze_points_3d = []
            ref_points_3d = []
            avg_error = 0.0
            avg_gaze_distance = 0.0
            avg_ref_distance = 0.0
            for i in range(0,len(ref_dir)):
                ref_p = ref_dir[i]
                gaze_direction = gaze_dir[i]
                gaze_direction_world = np.dot(rotation_matrix, gaze_direction)
                gaze_line = ( translation , translation + gaze_direction_world )
                ref_line = ( np.zeros(3) , ref_p )
                ref_point_world , gaze_point_world , distance = math_helper.nearest_intersection_points( ref_line , gaze_line )

                gaze_points_3d.append( gaze_point_world )
                ref_points_3d.append( ref_point_world )

                avg_error += distance
                avg_ref_distance += np.linalg.norm(ref_point_world)
                avg_gaze_distance += np.linalg.norm(gaze_point_world)


            avg_error /= len(ref_dir)
            avg_gaze_distance /= len(ref_dir)
            avg_ref_distance /= len(ref_dir)
            avg_distance = (avg_ref_distance + avg_gaze_distance) * 0.5
            logger.info('calibration average error: %s'%avg_error)
            logger.info('calibration average distance: %s'%avg_distance)

            print 'intersecting gaze and ref lines in space'
            print 'avg error: ' , avg_error
            print 'avg gaze distance: ' , avg_gaze_distance
            print 'avg ref distance: ' , avg_ref_distance

            g_pool.plugins.add(Vector_Gaze_Mapper,args=
                {'eye_camera_to_world_matrix':eye_camera_to_world_matrix , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_points_3d,
                 'cal_gaze_points_3d': gaze_points_3d,'gaze_distance':avg_distance})

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

