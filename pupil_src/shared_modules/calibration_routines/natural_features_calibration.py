'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from methods import normalize
from . finish_calibration import finish_calibration
from pyglui.cygl.utils import draw_points_norm,RGBA
from glfw import GLFW_PRESS
import audio


from pyglui import ui
from . calibration_plugin_base import Calibration_Plugin

# logging
import logging
logger = logging.getLogger(__name__)


class Natural_Features_Calibration(Calibration_Plugin):
    """Calibrate using natural features in a scene.
        Features are selected by a user by clicking on
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.first_img = None
        self.point = None
        self.count = 0
        self.detected = False
        self.pos = None
        self.r = 40.0  # radius of circle displayed
        self.ref_list = []
        self.pupil_list = []
        self.menu = None
        self.order = .5

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Manual Calibration"
        self.menu.append(ui.Info_Text("Calibrate gaze parameters using features in your environment. Ask the subject to look at objects in the scene and click on them in the world window."))

    def start(self):
        super().start()
        audio.say("Starting {}".format(self.mode_pretty))
        logger.info("Starting {}".format(self.mode_pretty))
        self.active = True
        self.ref_list = []
        self.pupil_list = []

    def stop(self):
        audio.say("Stopping  {}".format(self.mode_pretty))
        logger.info('Stopping  {}'.format(self.mode_pretty))
        self.active = False
        self.button.status_text = ''
        if self.mode == 'calibration':
            finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        if self.active:
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
                    nextPts = nextPts[0].tolist() #we prefer python types.
                    self.pos = normalize(nextPts,(gray.shape[1],gray.shape[0]),flip_y=True)
                    self.count -=1

                    ref = {}
                    ref["screen_pos"] = nextPts
                    ref["norm_pos"] = self.pos
                    ref["timestamp"] = frame.timestamp
                    self.ref_list.append(ref)

            # Always save pupil positions
            self.pupil_list.extend(events['pupil_positions'])

            if self.count:
                self.button.status_text = 'Sampling Gaze Data'
            else:
                self.button.status_text = 'Click to Sample at Location'

    def gl_display(self):
        if self.detected:
            draw_points_norm([self.pos], size=self.r, color=RGBA(0., 1., 0., .5))

    def on_click(self, pos, button, action):
        if action == GLFW_PRESS and self.active:
            self.first_img = None
            self.point = np.array([pos], dtype=np.float32)
            self.count = 30

    def get_init_dict(self):
        return {}

    def deinit_ui(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        super().deinit_ui()
