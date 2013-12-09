'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_points_norm
from plugin import Plugin
import numpy as np

from methods import denormalize

class Display_Gaze(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool,atb_pos=None):
        super(Display_Gaze, self).__init__()
        self.g_pool = g_pool
        self.atb_pos = atb_pos
        self.pupil_display_list = []

    def update(self,frame,recent_pupil_positions):
        for pt in recent_pupil_positions:
            if pt['norm_gaze'] is not None:
                self.pupil_display_list.append(pt['norm_gaze'])
        self.pupil_display_list[:-3] = []

    def gl_display(self,world_img_texture):
        draw_gl_points_norm(self.pupil_display_list,size=35,color=(1.,.2,.4,.6))


