
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

def finish_calibration(g_pool,pupil_list,ref_list,calibration_distance_3d = 500, force=None):
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

    matched_binocular_data = calibrate.closest_matches_binocular(ref_list,pupil_list)
    matched_pupil0_data = calibrate.closest_matches_monocular(ref_list,pupil0)
    matched_pupil1_data = calibrate.closest_matches_monocular(ref_list,pupil1)

    if len(matched_pupil0_data)>len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data
    logger.info('Collected %s monocular calibration data.'%len(matched_monocular_data))
    logger.info('Collected %s binocular calibration data.'%len(matched_binocular_data))



    if force:
        mode = force
    else:
        mode = g_pool.detection_mapping_mode

    if mode == '3d' and not camera_intrinsics:
        mode = '2d'
        logger.warning("Please calibrate your world camera using 'camera intrinsics estimation' for 3d gaze mapping.")



    if mode == '3d':
        if matched_binocular_data:
            method = 'binocular 3d model'
            cal_pt_cloud = calibrate.preprocess_3d_data_binocular_gaze_direction(matched_binocular_data,
                                            camera_intrinsics = camera_intrinsics,
                                            calibration_distance=calibration_distance_3d )
            cal_pt_cloud = np.array(cal_pt_cloud)
            try:
                gaze_direction0_3d = cal_pt_cloud[:,0]
                gaze_direction1_3d = cal_pt_cloud[:,1]
                ref_3d = cal_pt_cloud[:,2]
            except:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            sphere_pos0 = pupil0[-1]['sphere']['center']
            sphere_pos1 = pupil1[-1]['sphere']['center']
            initial_orientation0 = [ 0.05334223 , 0.93651217 , 0.07765971 ,-0.33774033] #eye0
            initial_orientation1 = [ 0.34200577 , 0.21628107 , 0.91189657 ,   0.06855066] #eye1
            initial_translation0 = (25, 15, -10)
            initial_translation1 = (-5, 15, -10)

            success0, orientation0, translation0  = point_line_calibration(sphere_pos0, ref_3d,  gaze_direction0_3d, initial_orientation0, initial_translation0)
            success1, orientation1, translation1  = point_line_calibration(sphere_pos1, ref_3d,  gaze_direction1_3d, initial_orientation1, initial_translation1)

            if not success0 or not success1:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            translation0 = np.ndarray(shape=(3,1), buffer=np.array(translation0))
            rotation_matrix0 = calibrate.quat2mat(orientation0)
            eye_to_world_matrix0  = np.matrix(np.eye(4))
            eye_to_world_matrix0[:3,:3] = rotation_matrix0
            eye_to_world_matrix0[:3,3:4] = translation0

            translation1 = np.ndarray(shape=(3,1), buffer=np.array(translation1))
            rotation_matrix1 = calibrate.quat2mat(orientation1)
            eye_to_world_matrix1  = np.matrix(np.eye(4))
            eye_to_world_matrix1[:3,:3] = rotation_matrix1
            eye_to_world_matrix1[:3,3:4] = translation1

            world_to_eye_matrix0  = np.linalg.inv(eye_to_world_matrix0)
            world_to_eye_matrix1  = np.linalg.inv(eye_to_world_matrix1)

            #with the eye_to_world matrix let's calculate the nearest points for every gaze line
            sphere_pos_world0 = np.zeros(4)
            sphere_pos_world0[:3] = sphere_pos0
            sphere_pos_world0[3] = 1.0
            sphere_pos_world0  = eye_to_world_matrix0.dot(sphere_pos_world0)
            sphere_pos_world0 = np.squeeze(np.asarray(sphere_pos_world0))

            sphere_pos_world1 = np.zeros(4)
            sphere_pos_world1[:3] = sphere_pos1
            sphere_pos_world1[3] = 1.0
            sphere_pos_world1  = eye_to_world_matrix1.dot(sphere_pos_world1)
            sphere_pos_world1 = np.squeeze(np.asarray(sphere_pos_world1))

            gaze_pt0_3d = []
            gaze_pt1_3d = []
            for i in range(0,len(ref_3d)):
                ref_p = ref_3d[i]
                gaze_direction0 = np.ones(3)
                gaze_direction0[:3] = gaze_direction0_3d[i]
                gaze_direction0 = rotation_matrix0.dot(gaze_direction0)
                gaze_direction0 = np.squeeze(np.asarray(gaze_direction0))

                gaze_direction1 = np.ones(3)
                gaze_direction1[:3] = gaze_direction1_3d[i]
                gaze_direction1 = rotation_matrix0.dot(gaze_direction1)
                gaze_direction1 = np.squeeze(np.asarray(gaze_direction1))

                line0 = (sphere_pos_world0[:3] , sphere_pos_world0[:3] + gaze_direction0[:3] )
                line1 = (sphere_pos_world1[:3] , sphere_pos_world1[:3] + gaze_direction1[:3] )

                point_world0 = calibrate.nearest_linepoint_to_point( ref_p , line0 )
                point_eye0 = np.ones(4)
                point_eye0[:3] = point_world0
                # everythings assumes gaze_points in eye coordinates
                point_eye0 = world_to_eye_matrix0.dot(point_eye0)
                point_eye0 = np.squeeze(np.asarray(point_eye0))
                gaze_pt0_3d.append( point_eye0[:3] )

                point_world1 = calibrate.nearest_linepoint_to_point( ref_p , line1 )
                point_eye1 = np.ones(4)
                point_eye1[:3] = point_world1
                # everythings assumes gaze_points in eye coordinates
                point_eye1 = world_to_eye_matrix0.dot(point_eye1)
                point_eye1 = np.squeeze(np.asarray(point_eye1))
                gaze_pt1_3d.append( point_eye1[:3] )


            # TODO restructure mapper and visualizer to handle gaze points in world coordinates
            avg_distance0, dist_var0 = calibrate.calculate_residual_3D_Points( ref_3d, gaze_pt0_3d, eye_to_world_matrix0 )
            avg_distance1, dist_var1 = calibrate.calculate_residual_3D_Points( ref_3d, gaze_pt1_3d, eye_to_world_matrix1 )
            logger.info('calibration average distance eye0: %s'%avg_distance0)
            logger.info('calibration average distance eye1: %s'%avg_distance1)

            g_pool.plugins.add(Binocular_Vector_Gaze_Mapper,args={'eye_to_world_matrix0':eye_to_world_matrix0,'eye_to_world_matrix1':eye_to_world_matrix1 , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_3d.tolist(), 'cal_gaze_points0_3d': gaze_pt0_3d, 'cal_gaze_points1_3d': gaze_pt1_3d })

        elif matched_monocular_data:
            method = 'monocular 3d model'
            cal_pt_cloud = calibrate.preprocess_3d_data_monocular_gaze_direction(matched_monocular_data,
                                            camera_intrinsics = camera_intrinsics,
                                            calibration_distance=calibration_distance_3d )
            cal_pt_cloud = np.array(cal_pt_cloud)
            try:
                gaze_direction_3d = cal_pt_cloud[:,0]
                ref_3d = cal_pt_cloud[:,1]
            except:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            sphere_pos = pupil0[-1]['sphere']['center']
            initial_orientation = [ 0.05334223 , 0.93651217 , 0.07765971 ,-0.33774033]
            #initial_orientation = [ 0.34200577 , 0.21628107 , 0.91189657 ,   0.06855066] #eye1
            initial_translation = (15, 30,-20)

            success, orientation, translation  = point_line_calibration(sphere_pos, ref_3d,  gaze_direction_3d, initial_orientation, initial_translation)

            if not success:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            translation = np.ndarray(shape=(3,1), buffer=np.array(translation))
            rotation_matrix = calibrate.quat2mat(orientation)
            eye_to_world_matrix  = np.matrix(np.eye(4))
            eye_to_world_matrix[:3,:3] = rotation_matrix
            eye_to_world_matrix[:3,3:4] = translation

            world_to_eye_matrix  = np.linalg.inv(eye_to_world_matrix)

            #with the eye_to_world matrix let's calculate the nearest points for every gaze line
            sphere_pos_world = np.zeros(4)
            sphere_pos_world[:3] = sphere_pos
            sphere_pos_world[3] = 1.0
            sphere_pos_world  = eye_to_world_matrix.dot(sphere_pos_world)
            sphere_pos_world = np.squeeze(np.asarray(sphere_pos_world))

            gaze_points_3d = []

            for i in range(0,len(ref_3d)):
                ref_p = ref_3d[i]
                gaze_direction = np.ones(3)
                gaze_direction[:3] = gaze_direction_3d[i]
                gaze_direction = rotation_matrix.dot(gaze_direction)
                gaze_direction = np.squeeze(np.asarray(gaze_direction))
                line = (sphere_pos_world[:3] , sphere_pos_world[:3] + gaze_direction[:3] )
                point_world = calibrate.nearest_linepoint_to_point( ref_p , line )
                point_eye = np.ones(4)
                point_eye[:3] = point_world
                # everythings assumes gaze_points in eye coordinates
                point_eye = world_to_eye_matrix.dot(point_eye)
                point_eye = np.squeeze(np.asarray(point_eye))
                gaze_points_3d.append( point_eye[:3] )

            # TODO restructure mapper and visualizer to handle gaze points in world coordinates
            avg_distance, dist_var = calibrate.calculate_residual_3D_Points( ref_3d, gaze_points_3d , eye_to_world_matrix  )
            logger.info('calibration average distance: %s'%avg_distance)

            g_pool.plugins.add(Vector_Gaze_Mapper,args={'eye_to_world_matrix':eye_to_world_matrix , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': cal_pt_cloud[:,1].tolist(), 'cal_gaze_points_3d': gaze_points_3d})

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



def finish_calibration_rays(g_pool,pupil_list,ref_list):
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
            cal_pt_cloud = calibrate.preprocess_3d_data_binocular_gaze_direction(matched_binocular_data,
                                            camera_intrinsics = camera_intrinsics,
                                            calibration_distance=1.0 )
            cal_pt_cloud = np.array(cal_pt_cloud)
            try:
                gaze_direction0_3d = cal_pt_cloud[:,0]
                gaze_direction1_3d = cal_pt_cloud[:,1]
                ref_3d = cal_pt_cloud[:,2]
            except:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            sphere_pos0 = pupil0[-1]['sphere']['center']
            sphere_pos1 = pupil1[-1]['sphere']['center']
            initial_orientation0 = [ 0.05334223 , 0.93651217 , 0.07765971 ,-0.33774033] #eye0
            initial_orientation1 = [ 0.34200577 , 0.21628107 , 0.91189657 ,   0.06855066] #eye1
            initial_translation0 = (25, 15, -10)
            initial_translation1 = (-5, 15, -10)

            success0, orientation0, translation0  = line_line_calibration(sphere_pos0, ref_3d,  gaze_direction0_3d, initial_orientation0, initial_translation0, fix_translation = True)
            success1, orientation1, translation1  = line_line_calibration(sphere_pos1, ref_3d,  gaze_direction1_3d, initial_orientation1, initial_translation1, fix_translation = True)

            if not success0 or not success1:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            translation0 = np.ndarray(shape=(3,1), buffer=np.array(translation0))
            rotation_matrix0 = calibrate.quat2mat(orientation0)
            eye_to_world_matrix0  = np.matrix(np.eye(4))
            eye_to_world_matrix0[:3,:3] = rotation_matrix0
            eye_to_world_matrix0[:3,3:4] = translation0

            translation1 = np.ndarray(shape=(3,1), buffer=np.array(translation1))
            rotation_matrix1 = calibrate.quat2mat(orientation1)
            eye_to_world_matrix1  = np.matrix(np.eye(4))
            eye_to_world_matrix1[:3,:3] = rotation_matrix1
            eye_to_world_matrix1[:3,3:4] = translation1

            world_to_eye_matrix0  = np.linalg.inv(eye_to_world_matrix0)
            world_to_eye_matrix1  = np.linalg.inv(eye_to_world_matrix1)

            #with the eye_to_world matrix let's calculate the nearest points for every gaze line
            sphere_pos_world0 = np.zeros(4)
            sphere_pos_world0[:3] = sphere_pos0
            sphere_pos_world0[3] = 1.0
            sphere_pos_world0  = eye_to_world_matrix0.dot(sphere_pos_world0)
            sphere_pos_world0 = np.squeeze(np.asarray(sphere_pos_world0))

            sphere_pos_world1 = np.zeros(4)
            sphere_pos_world1[:3] = sphere_pos1
            sphere_pos_world1[3] = 1.0
            sphere_pos_world1  = eye_to_world_matrix1.dot(sphere_pos_world1)
            sphere_pos_world1 = np.squeeze(np.asarray(sphere_pos_world1))

            gaze_pt0_3d = []
            gaze_pt1_3d = []
            for i in range(0,len(ref_3d)):
                ref_p = ref_3d[i]
                gaze_direction0 = np.ones(3)
                gaze_direction0[:3] = gaze_direction0_3d[i]
                gaze_direction0 = rotation_matrix0.dot(gaze_direction0)
                gaze_direction0 = np.squeeze(np.asarray(gaze_direction0))

                gaze_direction1 = np.ones(3)
                gaze_direction1[:3] = gaze_direction1_3d[i]
                gaze_direction1 = rotation_matrix0.dot(gaze_direction1)
                gaze_direction1 = np.squeeze(np.asarray(gaze_direction1))

                line0 = (sphere_pos_world0[:3] , sphere_pos_world0[:3] + gaze_direction0[:3] )
                line1 = (sphere_pos_world1[:3] , sphere_pos_world1[:3] + gaze_direction1[:3] )

                point_world0 = calibrate.nearest_linepoint_to_point( ref_p , line0 )
                point_eye0 = np.ones(4)
                point_eye0[:3] = point_world0
                # everythings assumes gaze_points in eye coordinates
                point_eye0 = world_to_eye_matrix0.dot(point_eye0)
                point_eye0 = np.squeeze(np.asarray(point_eye0))
                gaze_pt0_3d.append( point_eye0[:3] )

                point_world1 = calibrate.nearest_linepoint_to_point( ref_p , line1 )
                point_eye1 = np.ones(4)
                point_eye1[:3] = point_world1
                # everythings assumes gaze_points in eye coordinates
                point_eye1 = world_to_eye_matrix0.dot(point_eye1)
                point_eye1 = np.squeeze(np.asarray(point_eye1))
                gaze_pt1_3d.append( point_eye1[:3] )


            # TODO restructure mapper and visualizer to handle gaze points in world coordinates
            avg_distance0, dist_var0 = calibrate.calculate_residual_3D_Points( ref_3d, gaze_pt0_3d, eye_to_world_matrix0 )
            avg_distance1, dist_var1 = calibrate.calculate_residual_3D_Points( ref_3d, gaze_pt1_3d, eye_to_world_matrix1 )
            logger.info('calibration average distance eye0: %s'%avg_distance0)
            logger.info('calibration average distance eye1: %s'%avg_distance1)

            g_pool.plugins.add(Binocular_Vector_Gaze_Mapper,args={'eye_to_world_matrix0':eye_to_world_matrix0,'eye_to_world_matrix1':eye_to_world_matrix1 , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_3d.tolist(), 'cal_gaze_points0_3d': gaze_pt0_3d, 'cal_gaze_points1_3d': gaze_pt1_3d })

        elif matched_monocular_data:
            method = 'monocular 3d model'
            cal_pt_cloud = calibrate.preprocess_3d_data_monocular_gaze_direction(matched_monocular_data,
                                            camera_intrinsics = camera_intrinsics,
                                            calibration_distance=1.0 )
            cal_pt_cloud = np.array(cal_pt_cloud)
            try:
                gaze_direction_3d = cal_pt_cloud[:,0]
                ref_3d = cal_pt_cloud[:,1]
            except:
                logger.error(not_enough_data_error_msg)
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':not_enough_data_error_msg,'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            sphere_pos = matched_monocular_data[-1]['pupil']['sphere']['center']

            if matched_monocular_data[-1]['pupil']['id'] is 'eye0':
                initial_orientation = [ 0.05334223 , 0.93651217 , 0.07765971 ,-0.33774033] #eye0
                initial_translation = (25, 15, -10)
            else:
                initial_orientation = [ 0.34200577 , 0.21628107 , 0.91189657 ,   0.06855066] #eye1
                initial_translation = (-5, 15, -10)

            success, orientation, translation  = line_line_calibration(sphere_pos, ref_3d,  gaze_direction_3d, initial_orientation, initial_translation, fix_translation = True)
            print 'orientation: ' , orientation
            print 'translation: ' , translation

            if not success:
                logger.error("Calibration solver faild to converge.")
                g_pool.active_calibration_plugin.notify_all({'subject':'calibration_failed','reason':"Calibration solver faild to converge.",'timestamp':g_pool.capture.get_timestamp(),'record':True})
                return

            translation = np.matrix( translation )
            translation.shape = (3,1)
            rotation_matrix = calibrate.quat2mat(orientation)
            eye_to_world_matrix  = np.matrix(np.eye(4))
            eye_to_world_matrix[:3,:3] = rotation_matrix
            eye_to_world_matrix[:3,3:4] = translation
            print eye_to_world_matrix
            world_to_eye_matrix  = np.linalg.inv(eye_to_world_matrix)

            #with the eye_to_world matrix let's calculate the nearest points for every gaze line
            sphere_pos_world = np.zeros(4)
            sphere_pos_world[:3] = sphere_pos
            sphere_pos_world[3] = 1.0
            sphere_pos_world  = eye_to_world_matrix.dot(sphere_pos_world)
            sphere_pos_world = np.squeeze(np.asarray(sphere_pos_world))

            gaze_points_3d = []
            ref_points_3d = []

            for i in range(0,len(ref_3d)):
                ref_p = ref_3d[i]
                gaze_direction = np.ones(3)
                gaze_direction[:3] = gaze_direction_3d[i]
                gaze_direction = rotation_matrix.dot(gaze_direction)
                gaze_direction = np.squeeze(np.asarray(gaze_direction))
                gaze_line = (sphere_pos_world[:3] , sphere_pos_world[:3] + gaze_direction[:3] )
                ref_line = ( np.zeros(3) , ref_p )
                ref_point_world , gaze_point_world , distance = calibrate.nearest_intersection_points( ref_line , gaze_line )
                point_eye = np.ones(4)
                point_eye[:3] = gaze_point_world
                # everythings assumes gaze_points in eye coordinates
                point_eye = world_to_eye_matrix.dot(point_eye)
                point_eye = np.squeeze(np.asarray(point_eye))
                gaze_points_3d.append( point_eye[:3] )
                ref_points_3d.append( ref_point_world )

            # TODO restructure mapper and visualizer to handle gaze points in world coordinates
            avg_distance, dist_var = calibrate.calculate_residual_3D_Points( ref_points_3d, gaze_points_3d , eye_to_world_matrix  )
            logger.info('calibration average distance: %s'%avg_distance)

            g_pool.plugins.add(Vector_Gaze_Mapper,args={'eye_to_world_matrix':eye_to_world_matrix , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_points_3d, 'cal_gaze_points_3d': gaze_points_3d})

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

