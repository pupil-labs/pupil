
import os
import numpy as np

import calibrate
from file_methods import load_object,save_object
from camera_intrinsics_estimation import load_camera_calibration

#logging
import logging
logger = logging.getLogger(__name__)
from gaze_mappers import *

def finish_calibration(g_pool,pupil_list,ref_list,calibration_distance_3d = 500, force=None):
    if pupil_list and ref_list:
        pass
    else:
        logger.error("Did not collect calibration data. Please retry.")
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
            cal_pt_cloud = calibrate.preprocess_3d_data_binocular(matched_binocular_data,
                                        camera_intrinsics = camera_intrinsics,
                                        calibration_distance=calibration_distance_3d )
            cal_pt_cloud = np.array(cal_pt_cloud)
            try:
                gaze_pt0_3d = cal_pt_cloud[:,0]
                gaze_pt1_3d = cal_pt_cloud[:,1]
                ref_3d = cal_pt_cloud[:,2]
            except:
                logger.error('Did not collect enough data. Please retry.')
                return

            best_distance = 1000
            best_scale = 100
            for scale_percent in range(50,150,10):
                R0,t0 = calibrate.rigid_transform_3D( np.matrix(gaze_pt0_3d) ,np.matrix(ref_3d*(scale_percent/100.)) )
                R1,t1 = calibrate.rigid_transform_3D( np.matrix(gaze_pt1_3d) ,np.matrix(ref_3d*(scale_percent/100.)) )
                eye_to_world_matrix0  = np.matrix(np.eye(4))
                eye_to_world_matrix0[:3,:3] = R0
                eye_to_world_matrix0[:3,3:4] =  t0

                eye_to_world_matrix1  = np.matrix(np.eye(4))
                eye_to_world_matrix1[:3,:3] = R1
                eye_to_world_matrix1[:3,3:4] =  t1

                avg_distance0, dist_var0 = calibrate.calculate_residual_3D_Points( ref_3d*(scale_percent/100.), gaze_pt0_3d, eye_to_world_matrix0 )
                avg_distance1, dist_var1 = calibrate.calculate_residual_3D_Points( ref_3d*(scale_percent/100.), gaze_pt1_3d, eye_to_world_matrix1 )
                avg_distance = (avg_distance0+avg_distance1)/2.
                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_scale = scale_percent

            ref_3d *= best_scale/100.
            R0,t0 = calibrate.rigid_transform_3D( np.matrix(gaze_pt0_3d) ,np.matrix(ref_3d) )
            R1,t1 = calibrate.rigid_transform_3D( np.matrix(gaze_pt1_3d) ,np.matrix(ref_3d) )

            sphere0 = pupil0[-1]['sphere']['center']
            sphere1 = pupil1[-1]['sphere']['center']


            eye_to_world_matrix0  = np.matrix(np.eye(4))
            eye_to_world_matrix0[:3,:3] = R0
            eye_to_world_matrix0[:3,3:4] =  t0
            # eye_to_world_matrix0[:3,3:4] =  np.array((20,10,-20)).reshape(3,1)
            # eye_to_world_matrix0[:3,3:4] -=  R0 * (np.array(sphere0)*(1,-1,1)).reshape(3,1)

            eye_to_world_matrix1  = np.matrix(np.eye(4))
            eye_to_world_matrix1[:3,:3] = R1
            eye_to_world_matrix1[:3,3:4] =  t1
            # eye_to_world_matrix1[:3,3:4] =  np.array((-40,10,-20)).reshape(3,1)
            # eye_to_world_matrix1[:3,3:4] -=  R1 * (np.array(sphere1)*(1,-1,1)).reshape(3,1)

            avg_distance0, dist_var0 = calibrate.calculate_residual_3D_Points( ref_3d, gaze_pt0_3d, eye_to_world_matrix0 )
            avg_distance1, dist_var1 = calibrate.calculate_residual_3D_Points( ref_3d, gaze_pt1_3d, eye_to_world_matrix1 )
            logger.info('calibration average distance eye0: %s'%avg_distance0)
            logger.info('calibration average distance eye1: %s'%avg_distance1)

            g_pool.plugins.add(Binocular_Vector_Gaze_Mapper,args={'eye_to_world_matrix0':eye_to_world_matrix0,'eye_to_world_matrix1':eye_to_world_matrix1 , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': ref_3d.tolist(), 'cal_gaze_points0_3d': gaze_pt0_3d.tolist(), 'cal_gaze_points1_3d': gaze_pt1_3d.tolist() })

        elif matched_monocular_data:
            method = 'monocular 3d model'
            cal_pt_cloud = calibrate.preprocess_3d_data_monocular(matched_monocular_data,
                                            camera_intrinsics = camera_intrinsics,
                                            calibration_distance=calibration_distance_3d )
            cal_pt_cloud = np.array(cal_pt_cloud)
            try:
                gaze_3d = cal_pt_cloud[:,0]
                ref_3d = cal_pt_cloud[:,1]
            except:
                logger.error('Did not collect enough data. Please retry.')
                return

            best_distance = 1000
            best_scale = 100
            for scale_percent in range(50,150,10):
                #calculate transformation form eye camera to world camera
                R,t = calibrate.rigid_transform_3D( np.matrix(gaze_3d), np.matrix(ref_3d*(scale_percent/100.)) )

                eye_to_world_matrix  = np.matrix(np.eye(4))
                eye_to_world_matrix[:3,:3] = R
                eye_to_world_matrix[:3,3:4] = t

                avg_distance, dist_var = calibrate.calculate_residual_3D_Points( ref_3d*(scale_percent/100.), gaze_3d, eye_to_world_matrix )

                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_scale = scale_percent


            ref_3d *= best_scale/100.

            #calculate transformation form eye camera to world camera
            R,t = calibrate.rigid_transform_3D( np.matrix(gaze_3d), np.matrix(ref_3d) )

            eye_to_world_matrix  = np.matrix(np.eye(4))
            eye_to_world_matrix[:3,:3] = R
            eye_to_world_matrix[:3,3:4] = t
            avg_distance, dist_var = calibrate.calculate_residual_3D_Points( ref_3d, gaze_3d, eye_to_world_matrix )
            print 'best calibration average distance: ' , avg_distance
            # print 'best calibration distance variance: ' , dist_var

            g_pool.plugins.add(Vector_Gaze_Mapper,args={'eye_to_world_matrix':eye_to_world_matrix , 'camera_intrinsics': camera_intrinsics , 'cal_ref_points_3d': cal_pt_cloud[:,1].tolist(), 'cal_gaze_points_3d': cal_pt_cloud[:,0].tolist()})

        else:
            logger.error('Did not collect data during calibration.')
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
            logger.error('Did not collect data during calibration.')
            return

    user_calibration_data = {'pupil_list':pupil_list,'ref_list':ref_list,'calibration_method':method}
    save_object(user_calibration_data,os.path.join(g_pool.user_dir, "user_calibration_data"))
