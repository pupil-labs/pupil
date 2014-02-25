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
from scan_path import Scan_Path

logger = logging.getLogger(__name__)

class Filter_Fixations(Plugin):
    """docstring
    using this plugin will filter the recent_pupil_positions by manhattan distance from previous frame
    only recent_pupil_positions within distance tolerance will be shown
    """
    def __init__(self, g_pool=None,distance=25.0):
        super(Filter_Fixations, self).__init__()


        # let the plugin work after most other plugins
        self.order = .7

        # user settings
        self.distance = c_float(float(distance))

        # initialize dependencies
        # Scan_Path
        # check first if scanpath already exists
        # self.p_scan_path = Scan_Path(g_pool)
        # self.p_scan_path.timeframe.value = 1.0 # initialize wihout history
        # self.p_scan_path.init_gui()
        # g_pool.plugins.append(self.p_scan_path)
        # g_pool.plugins.sort(key=lambda p: p.order)


    def update(self,frame,recent_pupil_positions,events):
        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height  

        print "len recent: ", len(recent_pupil_positions)
        # setup to fire after scanpath
        # check if scan_path exists
        # compare distances of recent_pupil_positions list
        # check if scanpath is running -- if not then initialize it
        filtered_gaze = []

        for gp1, gp2 in zip(recent_pupil_positions[:-1], recent_pupil_positions[1:]):
            gp1_norm = denormalize(gp1['norm_gaze'], img_shape,flip_y=True)
            gp2_norm = denormalize(gp2['norm_gaze'], img_shape,flip_y=True)
            x_dist =  abs(gp1_norm[0] - gp2_norm[0])
            y_dist = abs(gp1_norm[1] - gp2_norm[1])
            man = x_dist + y_dist 
            # print "man: %s\tdist: %s" %(man,self.distance.value)
            if man < self.distance.value:
                filtered_gaze.append(gp1)
            else:
                print "filtered"

        print "filtered: ", len(filtered_gaze)
        recent_pupil_positions[:] = filtered_gaze
        recent_pupil_positions.sort(key=lambda x: x['timestamp']) #this may be redundant...            


        
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
        # self.p_scan_path.unset_alive()
        self._bar.destroy()



