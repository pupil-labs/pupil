'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

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
from ctypes import c_int,c_bool
from glfw import GLFW_PRESS
import atb
import audio

from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

class Natural_Features_Calibration(Plugin):
    """Calibrate using natural features in a scene.
        Features are selected by a user by clicking on
    """
    def __init__(self, g_pool,atb_pos=(0,0)):
        Plugin.__init__(self)
        self.g_pool = g_pool
        self.first_img = None
        self.point = None
        self.count = 0
        self.detected = False
        self.active = False
        self.pos = None
        self.r = 40.0 # radius of circle displayed
        self.ref_list = []
        self.pupil_list = []

        atb_label = "calibrate using natural features"
        self._bar = atb.Bar(name = self.__class__.__name__, label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("start/stop", self.start_stop, key='c')


    def start_stop(self):
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
        self.g_pool.map_pupil = calibrate.get_map_from_cloud(cal_pt_cloud,img_size)
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud.npy'),cal_pt_cloud)

    def update(self,frame,recent_pupil_positions,events):
        if self.active:
            img = frame.img
            if self.first_img is None:
                self.first_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            self.detected = False

            if self.count:
                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                # in cv2.3 nextPts is falsy required as an argument.
                nextPts_dummy = self.point.copy()
                nextPts,status, err = cv2.calcOpticalFlowPyrLK(self.first_img,gray,self.point,nextPts_dummy,winSize=(100,100))
                if status[0]:
                    self.detected = True
                    self.point = nextPts
                    self.first_img = gray
                    nextPts = nextPts[0]
                    self.pos = normalize(nextPts,(img.shape[1],img.shape[0]),flip_y=True)
                    self.count -=1

                    ref = {}
                    ref["norm_pos"] = self.pos
                    ref["timestamp"] = frame.timestamp
                    self.ref_list.append(ref)

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['norm_pupil'] is not None:
                    self.pupil_list.append(p_pt)

    def gl_display(self):
        if self.detected:
            draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))


    def on_click(self,pos,button,action):
        if action == GLFW_PRESS:
            self.first_img = None
            self.point = np.array([pos,],dtype=np.float32)
            self.count = 30

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        self._bar.destroy()