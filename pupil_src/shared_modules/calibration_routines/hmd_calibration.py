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
import audio
import numpy as np
from file_methods import save_object


from pyglui import ui
from . calibration_plugin_base import Calibration_Plugin
from . finish_calibration import not_enough_data_error_msg, solver_failed_to_converge_error_msg
from . import calibrate
from . gaze_mappers import Monocular_Gaze_Mapper, Dual_Monocular_Gaze_Mapper, Binocular_Vector_Gaze_Mapper
from . optimization_calibration import bundle_adjust_calibration
from . camera_intrinsics_estimation import idealized_camera_calibration
import math_helper

# logging
import logging
logger = logging.getLogger(__name__)


class HMD_Calibration(Calibration_Plugin):
    """Calibrate gaze on HMD screen.
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.info = None
        self.menu = None
        self.button = None

    def init_gui(self):

        def dummy(_):
            logger.error("HMD calibration must be initiated from the HMD client.")

        self.info = ui.Info_Text("Calibrate gaze parameters to map onto an HMD.")
        self.g_pool.calibration_menu.append(self.info)
        self.button = ui.Thumb('active',self,setter=dummy,label='C',hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)


    def on_notify(self,notification):
        '''Calibrates user gaze for HMDs

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``:  Stops the calibration procedure
           ``calibration.add_ref_data``: Adds reference data

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped

        Args:
            notification (dictionary): Notification dictionary
        '''
        try:
            if notification['subject'].startswith('calibration.should_start'):
                if self.active:
                    logger.warning('Calibration already running.')
                else:
                    hmd_video_frame_size = notification['hmd_video_frame_size']
                    outlier_threshold = notification['outlier_threshold']
                    self.start(hmd_video_frame_size,outlier_threshold)
            elif notification['subject'].startswith('calibration.should_stop'):
                if self.active:
                    self.stop()
                else:
                    logger.warning('Calibration already stopped.')
            elif notification['subject'].startswith('calibration.add_ref_data'):
                if self.active:
                    self.ref_list += notification['ref_data']
                else:
                    logger.error("Ref data can only be added when calibratio is runnings.")
        except KeyError as e:
            logger.error('Notification: {} not conform. Raised error {}'.format(notification,e))


    def deinit_gui(self):
        if self.info:
            self.g_pool.calibration_menu.remove(self.info)
            self.info = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def start(self,hmd_video_frame_size,outlier_threshold):
        self.active = True
        audio.say("Starting Calibration")
        logger.info("Starting Calibration")
        self.notify_all({'subject':'calibration.started'})
        self.pupil_list = []
        self.ref_list = []
        self.hmd_video_frame_size = hmd_video_frame_size
        self.outlier_threshold = outlier_threshold

    def stop(self):
        audio.say("Stopping Calibration")
        logger.info("Stopping Calibration")
        self.notify_all({'subject':'calibration.stopped'})
        self.active = False
        if self.button:
            self.button.status_text = ''

        pupil_list = self.pupil_list
        ref_list = self.ref_list
        hmd_video_frame_size = self.hmd_video_frame_size

        g_pool = self.g_pool

        pupil0 = [p for p in pupil_list if p['id']==0]
        pupil1 = [p for p in pupil_list if p['id']==1]

        ref0 = [r for r in ref_list if r['id']==0]
        ref1 = [r for r in ref_list if r['id']==1]

        matched_pupil0_data = calibrate.closest_matches_monocular(ref0,pupil0)
        matched_pupil1_data = calibrate.closest_matches_monocular(ref1,pupil1)

        if matched_pupil0_data:
            cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_pupil0_data)
            map_fn0,inliers0,params0 = calibrate.calibrate_2d_polynomial(cal_pt_cloud,hmd_video_frame_size,binocular=False)
            if not inliers0.any():
                self.notify_all({'subject':'calibration.failed','reason':solver_failed_to_converge_error_msg})
                return
        else:
            logger.warning('No matched ref<->pupil data collected for id0')
            params0 = None

        if matched_pupil1_data:
            cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_pupil1_data)
            map_fn1,inliers1,params1 = calibrate.calibrate_2d_polynomial(cal_pt_cloud,hmd_video_frame_size,binocular=False)
            if not inliers1.any():
                self.notify_all({'subject':'calibration.failed','reason':solver_failed_to_converge_error_msg})
                return
        else:
            logger.warning('No matched ref<->pupil data collected for id1')
            params1 = None

        if params0 and params1:
            g_pool.active_calibration_plugin.notify_all({'subject': 'start_plugin',
                                                         'name': 'Dual_Monocular_Gaze_Mapper',
                                                         'args': {'params0': params0,
                                                                  'params1': params1}})
            method = 'dual monocular polynomial regression'
        elif params0:
            g_pool.plugins.add(Monocular_Gaze_Mapper,args={'params':params0})
            g_pool.active_calibration_plugin.notify_all({'subject': 'start_plugin',
                                                         'name': 'Monocular_Gaze_Mapper',
                                                         'args': {'params': params0,
                                                                  }})
            method = 'monocular polynomial regression'
        elif params1:
            g_pool.active_calibration_plugin.notify_all({'subject': 'start_plugin',
                                                         'name': 'Monocular_Gaze_Mapper',
                                                         'args': {'params': params1,
                                                                  }})
            method = 'monocular polynomial regression'
        else:
            logger.error('Calibration failed for both eyes. No data found')
            self.notify_all({'subject':'calibration.failed','reason':not_enough_data_error_msg})
            return


    def recent_events(self,events):
        if self.active:
            for p_pt in events['pupil_positions']:
                if p_pt['confidence'] > self.pupil_confidence_threshold:
                    self.pupil_list.append(p_pt)

    def get_init_dict(self):
        d = {}
        return d

    def cleanup(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.active:
            self.stop()
        self.deinit_gui()


class HMD_Calibration_3D(HMD_Calibration,Calibration_Plugin):
    """docstring for HMD 3d calibratoin"""
    def __init__(self, g_pool):
        super(HMD_Calibration_3D, self).__init__(g_pool)

    def on_notify(self,notification):
        '''Calibrates user gaze for HMDs

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``:  Stops the calibration procedure
           ``calibration.add_ref_data``: Adds reference data

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped

        Args:
            notification (dictionary): Notification dictionary
        '''
        try:
            if notification['subject'].startswith('calibration.should_start'):
                if self.active:
                    logger.warning('Calibration already running.')
                else:
                    self.start()
            elif notification['subject'].startswith('calibration.should_stop'):
                if self.active:
                    self.stop()
                else:
                    logger.warning('Calibration already stopped.')
            elif notification['subject'].startswith('calibration.add_ref_data'):
                if self.active:
                    self.ref_list += notification['ref_data']
                else:
                    logger.error("Ref data can only be added when calibratio is running.")
        except KeyError as e:
            logger.error('Notification: %s not conform. Raised error %s'%(notification,e))

    def start(self):
        self.active = True
        audio.say("Starting Calibration")
        logger.info("Starting Calibration")
        self.notify_all({'subject':'calibration.started'})
        self.pupil_list = []
        self.ref_list = []

    def stop(self):
        audio.say("Stopping Calibration")
        logger.info("Stopping Calibration")
        self.notify_all({'subject':'calibration.stopped'})
        self.active = False
        if self.button:
            self.button.status_text = ''

        pupil_list = self.pupil_list
        ref_list = self.ref_list

        g_pool = self.g_pool

        not_enough_data_error_msg = 'Did not collect enough data during calibration.'
        solver_failed_to_converge_error_msg = 'Paramters could not be estimated from data.'

        matched_data = calibrate.closest_matches_binocular(ref_list,pupil_list)

        save_object(matched_data,'hmd_cal_data')


        ref_points_3d_unscaled = np.array([ d['ref']['mm_pos']      for d in matched_data ])
        gaze0_dir     = [ d['pupil']['circle_3d']['normal'] for d in matched_data ]
        gaze1_dir     = [ d['pupil1']['circle_3d']['normal']for d in matched_data ]

        if len(ref_points_3d_unscaled) < 1 or len(gaze0_dir) < 1 or len(gaze1_dir) < 1:
            logger.error(not_enough_data_error_msg)
            self.notify_all({'subject':'calibration.failed','reason':not_enough_data_error_msg,'timestamp':self.g_pool.get_timestamp(),'record':True})
            return

        smallest_residual = 1000
        scales = list(np.linspace(0.7,10,50))
        for s in scales:

            ref_points_3d = ref_points_3d_unscaled * (1,-1,s)



            initial_translation0 = np.array([30,0,0])
            initial_translation1 = np.array([-30,0,0])
            method = 'binocular 3d model hmd'

            sphere_pos0 = matched_data[-1]['pupil']['sphere']['center']
            sphere_pos1 = matched_data[-1]['pupil1']['sphere']['center']

            initial_R0,initial_t0 = calibrate.find_rigid_transform(np.array(gaze0_dir)*500,np.array(ref_points_3d)*1)
            initial_rotation0 = math_helper.quaternion_from_rotation_matrix(initial_R0)

            initial_R1,initial_t1 = calibrate.find_rigid_transform(np.array(gaze1_dir)*500,np.array(ref_points_3d)*1)
            initial_rotation1 = math_helper.quaternion_from_rotation_matrix(initial_R1)

            eye0 = { "observations" : gaze0_dir , "translation" : initial_translation0 , "rotation" : initial_rotation0,'fix':['translation']  }
            eye1 = { "observations" : gaze1_dir , "translation" : initial_translation1 , "rotation" : initial_rotation1,'fix':['translation']  }
            initial_observers = [eye0,eye1]
            initial_points = np.array(ref_points_3d)


            success, residual, observers, points = bundle_adjust_calibration(initial_observers , initial_points, fix_points=True )

            if residual <= smallest_residual:
                smallest_residual = residual
                scales[-1] = s

        if not success:
            self.notify_all({'subject':'calibration.failed','reason':solver_failed_to_converge_error_msg,'timestamp':self.g_pool.get_timestamp(),'record':True})
            logger.error("Calibration solver faild to converge.")
            return


        eye0, eye1 = observers

        t_world0 = np.array(eye0['translation'])
        R_world0 = math_helper.quaternion_rotation_matrix(np.array(eye0['rotation']))
        t_world1 = np.array(eye1['translation'])
        R_world1 = math_helper.quaternion_rotation_matrix(np.array(eye1['rotation']))

        def toWorld0(p):
            return np.dot(R_world0, p)+t_world0

        def toWorld1(p):
            return np.dot(R_world1, p)+t_world1

        points_a = points  # world coords
        points_b = []  # eye0 coords
        points_c = []  # eye1 coords

        for a, b, c, point in zip(points, eye0['observations'], eye1['observations'], points):
            line_a = np.array([0,0,0]) , np.array(a)  # observation as line
            line_b = toWorld0(np.array([0, 0, 0])), toWorld0(b)  # eye0 observation line in world coords
            line_c = toWorld1(np.array([0, 0,0 ])), toWorld1(c)  # eye1 observation line in world coords
            close_point_a, _ = math_helper.nearest_linepoint_to_point(point, line_a)
            close_point_b, _ = math_helper.nearest_linepoint_to_point(point, line_b)
            close_point_c, _ = math_helper.nearest_linepoint_to_point(point, line_c)
            points_a.append(close_point_a.tolist())
            points_b.append(close_point_b.tolist())
            points_c.append(close_point_c.tolist())

        # we need to take the sphere position into account
        # orientation and translation are referring to the sphere center.
        # but we want to have it referring to the camera center
        # since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates
        sphere_translation = np.array(sphere_pos0)
        sphere_translation_world = np.dot(R_world0, sphere_translation)
        camera_translation = t_world0 - sphere_translation_world
        eye_camera_to_world_matrix0 = np.eye(4)
        eye_camera_to_world_matrix0[:3, :3] = R_world0
        eye_camera_to_world_matrix0[:3, 3:4] = np.reshape(camera_translation, (3,1))

        sphere_translation = np.array(sphere_pos1)
        sphere_translation_world = np.dot(R_world1, sphere_translation)
        camera_translation = t_world1 - sphere_translation_world
        eye_camera_to_world_matrix1 = np.eye(4)
        eye_camera_to_world_matrix1[:3, :3] = R_world1
        eye_camera_to_world_matrix1[:3, 3:4] = np.reshape(camera_translation, (3, 1))

        method = 'binocular 3d model'
        ts = g_pool.get_timestamp()
        g_pool.active_calibration_plugin.notify_all({'subject': 'calibration.successful','method':method,'timestamp': ts, 'record':True})
        g_pool.active_calibration_plugin.notify_all({'subject': 'calibration.calibration_data','timestamp': ts, 'pupil_list':pupil_list,'ref_list':ref_list,'calibration_method':method,'record':True})

        # this is only used by show calibration. TODO: rewrite show calibraiton.
        user_calibration_data = {'timestamp': ts, 'pupil_list': pupil_list, 'ref_list': ref_list, 'calibration_method': method}
        save_object(user_calibration_data, os.path.join(g_pool.user_dir, "user_calibration_data"))

        scene_dummy_cam = idealized_camera_calibration((1280, 720), 700)
        self.g_pool.active_calibration_plugin.notify_all({'subject': 'start_plugin',
                                                         'name': 'Binocular_Vector_Gaze_Mapper',
                                                         'args': {'eye_camera_to_world_matrix0': eye_camera_to_world_matrix0.tolist(),
                                                                  'eye_camera_to_world_matrix1': eye_camera_to_world_matrix1.tolist(),
                                                                  'camera_intrinsics': scene_dummy_cam,
                                                                  'cal_points_3d': points,
                                                                  'cal_ref_points_3d': points_a,
                                                                  'cal_gaze_points0_3d': points_b,
                                                                  'cal_gaze_points1_3d': points_c}})
