'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from pyglui.cygl.utils import draw_points_norm,RGBA
from plugin import Plugin
import numpy as np

from methods import denormalize

class Display_Recent_Gaze(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool):
        super(Display_Recent_Gaze, self).__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []

    def update(self,frame,events):
        for pt in events.get('gaze',[]):
            self.pupil_display_list.append(pt['norm_pos'])
        self.pupil_display_list[:-3] = []


    def gl_display(self):
        draw_points_norm(self.pupil_display_list,size=35,color=RGBA(1.,.2,.4,.6))

    def get_init_dict(self):
        return {}
