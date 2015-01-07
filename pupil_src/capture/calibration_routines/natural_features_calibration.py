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
    def __init__(self, g_pool,menu_conf = {} ):
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


    def init_gui(self):

        self.menu = ui.Growing_Menu(self.pretty_class_name)
        self.menu.configuration = self.menu_conf
        self.g_pool.calibration_menu.append(self.menu)

        self.button = ui.Thumb('active',self,setter=self.toggle,label='Calibrate',hotkey='c')
        self.g_pool.quickbar.append(self.button)


    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.calibration_menu.remove(self.menu)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def toggle(self,new_var):
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
        cal_pt_cloud = calibrate.preprocess_data(self.pupil_list,self.ref_list)
        logger.info("Collected %s data points." %len(cal_pt_cloud))
        if len(cal_pt_cloud) < 20:
            logger.warning("Did not collect enough data.")
            return
        cal_pt_cloud = np.array(cal_pt_cloud)

        img_size = self.first_img.shape[1],self.first_img.shape[0]
        map_fn = calibrate.get_map_from_cloud(cal_pt_cloud,img_size)
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud.npy'),cal_pt_cloud)

        # prepare destruction of current gaze_mapper... and remove it
        for p in self.g_pool.plugins:
            if p.base_class_name == 'Gaze_Mapping_Plugin':
                p.alive = False
        self.g_pool.plugins = [p for p in g_pool.plugins if p.alive]

        #add new gaze mapper
        self.g_pool.plugins.append(Simple_Gaze_Mapper(self.g_pool,map_fn))
        self.g_pool.plugins.sort(key=lambda p: p.order)



    def update(self,frame,recent_pupil_positions,events):
        if self.active:
            img = frame.img
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
                    self.pos = normalize(nextPts,(img.shape[1],img.shape[0]),flip_y=True)
                    self.count -=1

                    ref = {}
                    ref["norm_pos"] = self.pos
                    ref["timestamp"] = frame.timestamp
                    self.ref_list.append(ref)

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['confidence'] > self.g_pool.pupil_confidence_threshold:
                    self.pupil_list.append(p_pt)

    def gl_display(self):
        if self.detected:
            draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))



    def on_click(self,pos,button,action):
        if action == GLFW_PRESS:
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
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        self.deinit_gui()