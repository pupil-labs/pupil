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
from methods import denormalize,normalize
from pyglui import ui
import logging
logger = logging.getLogger(__name__)

class Manual_Gaze_Correction(Plugin):
    """docstring
    correct gaze with manually set x and y offset
    """

    def __init__(self, g_pool,x_offset=0.,y_offset=0.,menu_conf={'pos':(10,390),'size':(300,100),'collapsed':False}):
        super(Manual_Gaze_Correction, self).__init__(g_pool)
        #let the plugin work before most other plugins.
        self.order = .3

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf
        #user settings
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)

    def update(self,frame,events):
        for p in events['pupil_positions']:
            if p['norm_gaze'] is not None:
                p['norm_gaze'] = p['norm_gaze'][0]+self.x_offset,p['norm_gaze'][1]+self.y_offset

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Manual Gaze Correction')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Info_Text('Move gaze horizontally and vertically. Screen width and height are one unit respectively.'))
        self.menu.append(ui.Slider('x_offset',self,min=-1,step=0.01,max=1))
        self.menu.append(ui.Slider('y_offset',self,min=-1,step=0.01,max=1))
        self.menu.append(ui.Button('remove',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'x_offset':self.x_offset,'y_offset':self.y_offset,'menu_conf':self.menu.configuration}


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

