
import os
import numpy as np

import calibrate
from file_methods import load_object,save_object
#logging
import logging
logger = logging.getLogger(__name__)
from gaze_mappers import *

def finish_calibration(g_pool,pupil_list,ref_list,calibration_distance_3d = 500):
    try:
        camera_calibration = load_object(os.path.join(g_pool.user_dir,'camera_calibration'))
    except IOError:
        camera_intrinsics = None
        logger.warning('No camera calibration.')
    else:
        same_name = camera_calibration['camera_name'] == g_pool.capture.name
        same_resolution =  camera_calibration['resolution'] == g_pool.capture.frame_size
        if same_name and same_resolution:
            logger.info('Loaded camera calibration. 3D marker tracking enabled.')
            K = camera_calibration['camera_matrix']
            dist_coefs = camera_calibration['dist_coefs']
            resolution = camera_calibration['resolution']
            camera_intrinsics = K,dist_coefs,resolution
        else:
            logger.info('Loaded camera calibration but camera name and/or resolution has changed. Please re-calibrate.')
            camera_intrinsics = None


    # do we have data from 3D detector?
    if camera_intrinsics and pupil_list[0] and pupil_list[0]['method'] == '3D c++':
        use_3d = True
    else:
        use_3d = False


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


    if use_3d:
        if matched_binocular_data:
            method = 'binocular 3d model'
            logger.error("Notimplemented")
        elif matched_monocular_data:
            method = 'monocular 3d model'
            cal_pt_cloud = calibrate.preprocess_3d_data_monocular(matched_monocular_data,
                                            camera_intrinsics = camera_intrinsics,
                                            calibration_distance=calibration_distance_3d)
            cal_pt_cloud = np.array(cal_pt_cloud)
            gaze_3d = cal_pt_cloud[:,0]
            ref_3d = cal_pt_cloud[:,1]
            print 'gaze: ' , gaze_3d
            print 'ref points: ' , ref_3d
            R,t = calibrate.rigid_transform_3D( np.matrix(gaze_3d), np.matrix(ref_3d) )
            transformation = cv2.Rodrigues( R)[0] , t
            print 'transformation: ' , transformation
            g_pool.plugins.add(Vector_Gaze_Mapper,args={'transformation':transformation , 'camera_intrinsics': camera_intrinsics , 'calibration_points_3d': cal_pt_cloud[:,0].tolist(), 'calibration_points_2d': cal_pt_cloud[:,1].tolist()})
        else:
            logger.error('Did not collect data during calibration.')

    else:
        if matched_binocular_data:
            method = 'binocular polynomial regression'
            cal_pt_cloud_binocular = calibrate.preprocess_2d_data_binocular(matched_binocular_data)
            cal_pt_cloud0 = calibrate.preprocess_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = calibrate.preprocess_2d_data_monocular(matched_pupil1_data)
            map_fn,inliers,params = calibrate.calibrate_2d_polynomial(cal_pt_cloud_binocular,g_pool.capture.frame_size,binocular=True)
            map_fn,inliers,params_eye0 = calibrate.calibrate_2d_polynomial(cal_pt_cloud0,g_pool.capture.frame_size,binocular=False)
            map_fn,inliers,params_eye1 = calibrate.calibrate_2d_polynomial(cal_pt_cloud1,g_pool.capture.frame_size,binocular=False)
            g_pool.plugins.add(Bilateral_Gaze_Mapper,args={'params':params, 'params_eye0':params_eye0, 'params_eye1':params_eye1})


        elif matched_monocular_data:
            method = 'monocular polynomial regression'
            cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_monocular_data)
            map_fn,inliers,params = calibrate.calibrate_2d_polynomial(cal_pt_cloud,g_pool.capture.frame_size,binocular=False)
            g_pool.plugins.add(Simple_Gaze_Mapper,args={'params':params})
        else:
            logger.error('Did not collect data during calibration.')


    user_calibration_data = {'pupil_list':pupil_list,'ref_list':ref_list,'calibration_method':method}
    save_object(user_calibration_data,os.path.join(g_pool.user_dir, "user_calibration_data"))
