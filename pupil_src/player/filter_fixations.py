'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np
import atb
from ctypes import c_float
from methods import denormalize,normalize
import logging
logger = logging.getLogger(__name__)

class Filter_Fixations(Plugin):
    """docstring
    using this plugin will filter the recent_pupil_positions by manhattan distance from previous frame
    only recent_pupil_positions within distance tolerance will be shown
    """
    def __init__(self, g_pool=None,distance=25.0):
        super(Filter_Fixations, self).__init__()


        # let the plugin work after most other plugins
        self.order = .6

        # user settings
        self.distance = c_float(float(distance))

        # algorithm working data
        self.prev_frame_idx = -1
        self.past_pupil_positions = []

    def update(self,frame,recent_pupil_positions,events):
        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        succeeding_frame = frame.index-self.prev_frame_idx == 1

        if self.past_pupil_positions and succeeding_frame:
            if recent_pupil_positions:
                now = denormalize(recent_pupil_positions[0]['norm_gaze'],img_shape,flip_y=True)
                previous = denormalize(self.past_pupil_positions[-1]['norm_gaze'],img_shape,flip_y=True)
                x_dist = abs(previous[0]-now[0])
                y_dist = abs(previous[1]-now[1])
                man = x_dist + y_dist
                if man < self.distance:
                    recent_pupil_positions[:] = recent_pupil_positions

        self.prev_frame_idx = frame.index
        self.past_pupil_positions = recent_pupil_positions

    def init_gui(self,pos=None):
        pos = 10,470
        import atb
        from time import time

        atb_label = "Filter Fixations"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="polyline", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.1, size=(300, 70))

        self._bar.add_var('distance in pixels',self.distance,min=0,step=0.1)

        self._bar.add_button('remove',self.unset_alive)

    def unset_alive(self):
        self.alive = False


    def get_init_dict(self):
        return {'distance':self.distance.value}


    def clone(self):
        return Filter_Fixations(**self.get_init_dict())


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()



