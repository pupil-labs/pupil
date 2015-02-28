'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from time import sleep
import numpy as np
from methods import *

import logging
logger = logging.getLogger(__name__)

# gui
from pyglui import ui

class Pupil_Detector(object):
    """base class for pupil detector"""
    def __init__(self,g_pool):
        super(Pupil_Detector, self).__init__()
        self.g_pool = g_pool
        
    def detect(self,frame,u_roi,visualize=False):
        img = frame.img
        # hint: create a view into the img with the bounds of user set region of interest
        pupil_img = img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX]

        if visualize:
            pass
            # draw into image whatever you like and it will be displayed
            # otherwise you shall not modify img data inplace!


        candidate_pupil_ellipse = {'center': (None,None),
                        'axes': (None, None),
                        'angle': None,
                        'area': None,
                        'ratio': None,
                        'major': None,
                        'minor': None,
                        'goodness': 0} #some estimation on how sure you are about the detected ellipse and its fit. Smaller is better

        # If you use region of interest p_roi and u_roi make sure to return pupil coordinates relative to the full image
        candidate_pupil_ellipse['center'] = u_roi.add_vector(candidate_pupil_ellipse['center'])
        candidate_pupil_ellipse['timestamp'] = frame.timestamp
        result = candidate_pupil_ellipse #we found something
        if result:
            return candidate_pupil_ellipse # all this will be sent to the world process, you can add whateever you need to this.

        else:
            self.goodness.value = 100
            no_result = {}
            no_result['timestamp'] = frame.timestamp
            no_result['norm_pupil'] = None
            return no_result

    def init_gui(self):
        pass






