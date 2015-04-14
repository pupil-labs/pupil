'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np
from pyglui import ui
from methods import denormalize,normalize
import logging
from scan_path import Scan_Path

logger = logging.getLogger(__name__)

class Filter_Fixations(Plugin):
    """docstring
    using this plugin will filter the recent_pupil_positions by manhattan distance from previous frame
    only recent_pupil_positions within distance tolerance will be shown
    """
    def __init__(self, g_pool=None,distance=25.0,menu_conf={'pos':(10,470),'size':(300,100),'collapsed':False}):
        super(Filter_Fixations, self).__init__(g_pool)
        self.g_pool = g_pool
        # let the plugin work after most other plugins
        self.order = .7

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf

        # user settings
        self.distance = distance
        self.sp_active = True

    def update(self,frame,events):
        # TODO: leave this to a dependency plugin loader
        if any(isinstance(p,Scan_Path) for p in self.g_pool.plugins):
            if self.sp_active:
                pass
            else:
                self.set_bar_ok(True)
                self.sp_active = True
        else:
            if self.sp_active:
                self.set_bar_ok(False)
                self.sp_active = False
            else:
                pass

        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        filtered_gaze = []

        for gp1, gp2 in zip(events['pupil_positions'][:-1], events['pupil_positions'][1:]):
            gp1_norm = denormalize(gp1['norm_gaze'], img_shape,flip_y=True)
            gp2_norm = denormalize(gp2['norm_gaze'], img_shape,flip_y=True)
            x_dist =  abs(gp1_norm[0] - gp2_norm[0])
            y_dist = abs(gp1_norm[1] - gp2_norm[1])
            man = x_dist + y_dist
            # print "man: %s\tdist: %s" %(man,self.distance)
            if man < self.distance:
                filtered_gaze.append(gp1)
            else:
                # print "filtered"
                pass

        events['pupil_positions'][:] = filtered_gaze[:]
        events['pupil_positions'].sort(key=lambda x: x['timestamp']) #this may be redundant...

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Filter Fixations')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Info_Text("Filter Fixations uses Scan_Path to understand past gaze"))
        self.menu.append(ui.Slider('distance',self,min=0,step=1,max=100,label='distance in pixels'))
        self.menu.append(ui.Button('remove',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def set_bar_ok(self,ok):
        if ok:
            self.menu[0].text = "Filter Fixations uses Scan_Path to understand past gaze"
        else:
            self.menu[0].text  = "Filter Fixations: Turn on Scan_Path!"

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'distance':self.distance, 'menu_conf':self.menu.configuration}


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()



