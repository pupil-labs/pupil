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

from methods import denormalize

class Display_Gaze(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool):
        super(Display_Gaze, self).__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []

    def update(self,frame,events):
        self.pupil_display_list = [pt['norm_gaze'] for pt in events['pupil_positions'] if pt['norm_gaze'] is not None]

    def gl_display(self):
        draw_gl_points_norm(self.pupil_display_list,size=35,color=(1.,.2,.4,.6))

