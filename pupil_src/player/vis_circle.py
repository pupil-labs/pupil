'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_points_norm
from player_methods import transparent_cirlce
from plugin import Plugin
import numpy as np
from ctypes import c_int,c_float,c_bool
import cv2

from methods import denormalize

class Vis_Circle(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool=None,radius=20,color=(1.,.2,.4,.5),thickness=1,fill=False):
        super(Vis_Circle, self).__init__()
        self.g_pool = g_pool
        self.order = .9


        self.radius = c_int(int(radius))
        self.color = (c_float*4)(*color)
        self.thickness = c_int(int(thickness))
        self.fill = c_bool(bool(fill))


    def update(self,frame,recent_pupil_positions,events):
        color = map(lambda x:int(x*255),self.color)
        color = color[:3][::-1]+color[-1:]
        if self.fill.value:
            thickness= -1
        else:
            thickness = self.thickness.value

        radius = self.radius.value
        pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]
        for pt in pts:
            transparent_cirlce(frame.img, pt, radius=radius, color=color, thickness=thickness)

    def init_gui(self,pos=None):
        pos = 10,200
        import atb
        from time import time
        atb_label = "Gaze Circle"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="circle", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.1, size=(300, 100))

        self._bar.add_var('color',self.color)
        self._bar.add_var('radius',self.radius, min=1)
        self._bar.add_var('thickness',self.thickness,min=1)
        self._bar.add_var('fill',self.fill)
        self._bar.add_button('remove',self.unset_alive)

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        pass

    def get_init_dict(self):
        return {'radius':self.radius.value,'color':self.color[:],'thickness':self.thickness.value,'fill':self.fill.value}

    def clone(self):
        return Vis_Circle(**self.get_init_dict())


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()


