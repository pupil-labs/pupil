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
import atb
from ctypes import c_float
from methods import denormalize,normalize
import logging
logger = logging.getLogger(__name__)

class Manual_Gaze_Correction(Plugin):
    """docstring
    correct gaze with manually set x and y offset

    """

    def __init__(self, g_pool=None,x_offset=0,y_offset=0,gui_settings={'pos':(10,390),'size':(300,100),'iconified':False}):
        super(Manual_Gaze_Correction, self).__init__()

        #let the plugin work before most other plugins.
        self.order = .3

        #user settings
        self.x_offset = c_float(float(x_offset))
        self.y_offset = c_float(float(y_offset))
        self.gui_settings = gui_settings



    def update(self,frame,recent_pupil_positions,events):
      for p in recent_pupil_positions:
        if p['norm_gaze'] is not None:
            p['norm_gaze'] = p['norm_gaze'][0]+self.x_offset.value,p['norm_gaze'][1]+self.y_offset.value


    def init_gui(self,pos=None):
        pos = self.gui_settings['pos']
        import atb
        atb_label = "Manual Gaze Correction"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="polyline", color=(50, 50, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])
        self._bar.iconified = self.gui_settings['iconified']
        self._bar.add_var('x_offset',self.x_offset,step=0.002, help="move gaze sideways. screen width is one unit")
        self._bar.add_var('y_offset',self.y_offset,step=0.002, help="move gaze up and down. screen height is one unit")
        self._bar.add_button("remove",self.unset_alive)


    def unset_alive(self):
        self.alive = False


    def get_init_dict(self):
        d = {'x_offset':self.x_offset.value,'y_offset':self.y_offset.value}

        if hasattr(self,'_bar'):
            gui_settings = {'pos':self._bar.position,'size':self._bar.size,'iconified':self._bar.iconified}
            d['gui_settings'] = gui_settings

        return d


    def clone(self):
        return Scan_Path(**self.get_init_dict())


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()

