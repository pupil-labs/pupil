'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_points_norm
from plugin import Plugin
import numpy as np

import cv2

from pyglui import ui
from methods import denormalize

class Vis_Polyline(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool,color=(1.0,0.0,0.4,1.0),thickness=2,menu_conf={'pos':(10,320),'size':(300,70),'collapsed':False}):
        super(Vis_Polyline, self).__init__(g_pool)
        self.order = .9
        self.uniqueness = "not_unique"

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf

        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]
        self.thickness = thickness

    def update(self,frame,events):
        pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events['pupil_positions'] if pt['norm_gaze'] is not None]
        bgra = (self.b*255,self.g*255,self.r*255,self.a*255)
        if pts:
            pts = np.array([pts],dtype=np.int32)
            cv2.polylines(frame.img, pts, isClosed=False, color=bgra, thickness=self.thickness, lineType=cv2.cv.CV_AA)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Gaze Polyline')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Slider('thickness',self,min=1,step=1,max=15,label='Line thickness'))

        color_menu = ui.Growing_Menu('Color')
        color_menu.collapsed = True
        color_menu.append(ui.Info_Text('Set RGB color component values.'))
        color_menu.append(ui.Slider('r',self,min=0.0,step=0.05,max=1.0,label='Red'))
        color_menu.append(ui.Slider('g',self,min=0.0,step=0.05,max=1.0,label='Green'))
        color_menu.append(ui.Slider('b',self,min=0.0,step=0.05,max=1.0,label='Blue'))
        self.menu.append(color_menu)

        self.menu.append(ui.Button('remove',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'color':(self.r, self.g, self.b, self.a),'thickness':self.thickness, 'menu_conf':self.menu.configuration}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
