'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from methods import normalize
import calibrate
from gl_utils import draw_gl_point_norm
from glfw import GLFW_PRESS
import audio


from pyglui import ui
from plugin import Calibration_Plugin
from gaze_mappers import Simple_Gaze_Mapper

#logging
import logging
logger = logging.getLogger(__name__)

class Natural_Features_Calibration(Calibration_Plugin):
    """Calibrate using natural features in a scene.
        Features are selected by a user by clicking on
    """
    def __init__(self, g_pool,menu_conf = {'collapsed':True} ):
        super(Natural_Features_Calibration, self).__init__(g_pool)
        self.first_img = None
        self.point = None
        self.count = 0
        self.detected = False
        self.active = False
        self.pos = None
        self.r = 40.0 # radius of circle displayed
        self.ref_list = []
        self.pupil_list = []


        self.menu = None
        self.menu_conf = menu_conf
        self.button = None

        self.order = .5


    def init_gui(self):

        self.info = ui.Info_Text("Calibrate gaze parameters using features in your environment. Ask the subject to look at objects in the scene and click on them in the world window.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.menu.configuration = self.menu_conf
        self.g_pool.calibration_menu.append(self.menu)

        self.button = ui.Thumb('active',self,setter=self.toggle,label='Calibrate',hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)


    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.calibration_menu.remove(self.menu)
            self.g_pool.calibration_menu.remove(self.info)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def toggle(self,_=None):
        if self.active:
            self.stop()
        else:
            self.start()



    def start(self):
        audio.say("Starting Calibration")
        logger.info("Starting Calibration")
        self.active = True
        self.ref_list = []
        self.pupil_list = []

    def stop(self):
        audio.say("Stopping Calibration")
        logger.info("Stopping Calibration")
        self.active = False
        self.button.status_text = ''

        cal_pt_cloud = calibrate.preprocess_data(self.pupil_list,self.ref_list)
        logger.info("Collected %s data points." %len(cal_pt_cloud))
        if len(cal_pt_cloud) < 20:
            logger.warning("Did not collect enough data.")
            return
        cal_pt_cloud = np.array(cal_pt_cloud)

        img_size = self.first_img.shape[1],self.first_img.shape[0]
        map_fn,params = calibrate.get_map_from_cloud(cal_pt_cloud,img_size,return_params=True)
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud.npy'),cal_pt_cloud)

        #replace gaze mapper
        self.g_pool.plugins.add(Simple_Gaze_Mapper(self.g_pool,params))



    def update(self,frame,events):
        if self.active:
            recent_pupil_positions = events['pupil_positions']

            if self.first_img is None:
                self.first_img = frame.gray.copy()

            self.detected = False

            if self.count:
                gray = frame.gray
                # in cv2.3 nextPts is falsly required as an argument.
                nextPts_dummy = self.point.copy()
                nextPts,status, err = cv2.calcOpticalFlowPyrLK(self.first_img,gray,self.point,nextPts_dummy,winSize=(100,100))
                if status[0]:
                    self.detected = True
                    self.point = nextPts
                    self.first_img = gray.copy()
                    nextPts = nextPts[0]
                    self.pos = normalize(nextPts,(gray.shape[1],gray.shape[0]),flip_y=True)
                    self.count -=1

                    ref = {}
                    ref["norm_pos"] = self.pos
                    ref["timestamp"] = frame.timestamp
                    self.ref_list.append(ref)

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['confidence'] > self.g_pool.pupil_confidence_threshold:
                    self.pupil_list.append(p_pt)

            if self.count:
                self.button.status_text = 'Sampling Gaze Data'
            else:
                self.button.status_text = 'Click to Sample at Location'




    def gl_display(self):
        if self.detected:
            draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))



    def on_click(self,pos,button,action):
        if action == GLFW_PRESS and self.active:
            self.first_img = None
            self.point = np.array([pos,],dtype=np.float32)
            self.count = 30

    def get_init_dict(self):
        if self.menu:
            return {'menu_conf':self.menu.configuration}
        else:
            return {'menu_conf':self.menu_conf}


    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        self.deinit_gui()