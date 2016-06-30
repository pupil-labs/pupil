'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from methods import normalize,denormalize
import audio

from pyglui import ui
from calibration_plugin_base import Calibration_Plugin
from finish_calibration import not_enough_data_error_msg,solver_failed_to_converge_error_msg
import calibrate
from gaze_mappers import Monocular_Gaze_Mapper,Dual_Monocular_Gaze_Mapper

#logging
import logging
logger = logging.getLogger(__name__)


class HMD_Calibration(Calibration_Plugin):
    """Calibrate gaze on HMD screen.
    """
    def __init__(self, g_pool):
        super(HMD_Calibration, self).__init__(g_pool)
        self.info = None
        self.menu = None
        self.button = None

    def init_gui(self):

        def dummy(_):
            logger.error("HMD calibration must be initiated from the HMD client.")

        self.info = ui.Info_Text("Calibrate gaze parameters to map onto an HMD.")
        self.g_pool.calibration_menu.append(self.info)
        self.button = ui.Thumb('active',self,setter=dummy,label='Calibrate',hotkey='c')
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
            logger.error('Notification: %s not conform. Raised error %s'%(notification,e))


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
            g_pool.plugins.add(Dual_Monocular_Gaze_Mapper,args={'params0':params0,'params1':params1})
            method = 'dual monocular polynomial regression'
        elif params0:
            g_pool.plugins.add(Monocular_Gaze_Mapper,args={'params':params0})
            method = 'monocular polynomial regression'
        elif params1:
            g_pool.plugins.add(Monocular_Gaze_Mapper,args={'params':params1})
            method = 'monocular polynomial regression'
        else:
            logger.error('Calibration failed for both eyes. No data found')
            self.notify_all({'subject':'calibration.failed','reason':not_enough_data_error_msg})
            return

        self.notify_all({'subject':'calibration.successful','method':method})


    def update(self,frame,events):
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
