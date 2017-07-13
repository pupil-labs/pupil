'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from player_methods import transparent_circle
from plugin import Visualizer_Plugin_Base
import numpy as np
import cv2
from methods import denormalize
from pyglui import ui


class Vis_Fixation(Visualizer_Plugin_Base):
    uniqueness = "not_unique"

    def __init__(self, g_pool,radius=20,color=(0.0,0.7,0.25,0.2),thickness=2,fill=True):
        super().__init__(g_pool)
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

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        if self.fill:
            thickness = -1
        else:
            thickness = self.thickness

        fixation_pts = [denormalize(pt['norm_pos'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events.get('fixations',[])]
        not_fixation_pts = [denormalize(pt['norm_pos'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events.get('gaze_positions',[])]


        if fixation_pts:
            for pt in fixation_pts:
                transparent_circle(frame.img, pt, radius=self.radius, color=(self.b, self.g, self.r, self.a), thickness=thickness)
        else:
            for pt in not_fixation_pts:
                transparent_circle(frame.img, pt, radius=7.0, color=(0.2, 0.0, 0.7, 0.5), thickness=thickness)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Fixation Circle')
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
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
