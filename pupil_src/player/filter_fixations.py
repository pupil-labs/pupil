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
    def __init__(self, g_pool=None,distance=25.0,gui_settings={'pos':(10,470),'size':(300,100),'iconified':False}):
        super(Filter_Fixations, self).__init__()


        # let the plugin work after most other plugins
        self.order = .7

        # user settings
        self.distance = c_float(float(distance))
        self.gui_settings = gui_settings

        # initialize dependencies
        # init Scan_Path if not already initialized
        self.sp_user = False

        for p in g_pool.plugins:
            if isinstance(p,Scan_Path):
                self.sp_user = True

        if not self.sp_user:
            # add scanpath
            self.p_scan_path = Scan_Path(g_pool)
            self.p_scan_path.timeframe.value = 1.0
            self.p_scan_path.init_gui()
            g_pool.plugins.append(self.p_scan_path)
            g_pool.plugins.sort(key=lambda p: p.order)


    def update(self,frame,recent_pupil_positions,events):
        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

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
                # print "filtered"
                pass

        recent_pupil_positions[:] = filtered_gaze[:]
        recent_pupil_positions.sort(key=lambda x: x['timestamp']) #this may be redundant...



    def init_gui(self,pos=None):
        import atb
        pos = self.gui_settings['pos']
        atb_label = "Filter Fixations"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="polyline", color=(50, 50, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])

        self._bar.iconified = self.gui_settings['iconified']
        self._bar.add_var('distance in pixels',self.distance,min=0,step=0.1)
        self._bar.add_button('remove',self.unset_alive)

    def unset_alive(self):
        self.alive = False


    def get_init_dict(self):
        d = {'distance':self.distance.value}
        if hasattr(self,'_bar'):
            gui_settings = {'pos':self._bar.position,'size':self._bar.size,'iconified':self._bar.iconified}
            d['gui_settings'] = gui_settings
        return d


    def clone(self):
        return Filter_Fixations(**self.get_init_dict())


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if not self.sp_user:
            self.p_scan_path.unset_alive()
        self._bar.destroy()



