'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from player_methods import transparent_circle
from plugin import Plugin
import numpy as np
import cv2

# TODO: Import pyglui
from pyglui import ui

from methods import denormalize

class Vis_Circle(Plugin):
    uniqueness = "not_unique"

    def __init__(self, g_pool,radius=20,color=(0.0,0.7,0.25,0.2),thickness=2,fill=True):
        super(Vis_Circle, self).__init__(g_pool)
        self.order = .9

        # initialize empty menu
        self.menu = None

        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]
        self.radius = radius
        self.thickness = thickness
        self.fill = fill

    def update(self,frame,events):
        if self.fill:
            thickness = -1
        else:
            thickness = self.thickness

        pts = [denormalize(pt['norm_pos'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events.get('gaze_positions',[])]
        for pt in pts:
            transparent_circle(frame.img, pt, radius=self.radius, color=(self.b, self.g, self.r, self.a), thickness=thickness)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Gaze Circle')
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Slider('radius',self,min=1,step=1,max=100,label='Radius'))
        self.menu.append(ui.Slider('thickness',self,min=1,step=1,max=15,label='Stroke width'))
        self.menu.append(ui.Switch('fill',self,label='Fill'))

        color_menu = ui.Growing_Menu('Color')
        color_menu.collapsed = True
        color_menu.append(ui.Info_Text('Set RGB color components and alpha (opacity) values.'))
        color_menu.append(ui.Slider('r',self,min=0.0,step=0.05,max=1.0,label='Red'))
        color_menu.append(ui.Slider('g',self,min=0.0,step=0.05,max=1.0,label='Green'))
        color_menu.append(ui.Slider('b',self,min=0.0,step=0.05,max=1.0,label='Blue'))
        color_menu.append(ui.Slider('a',self,min=0.0,step=0.05,max=1.0,label='Alpha'))
        self.menu.append(color_menu)

        self.menu.append(ui.Button('remove',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        pass

    def get_init_dict(self):
        return {'radius':self.radius,'color':(self.r, self.g, self.b, self.a),'thickness':self.thickness,'fill':self.fill}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()


