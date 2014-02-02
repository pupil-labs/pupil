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

import cv2

from methods import denormalize

class Vis_Polyline(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool):
        super(Vis_Polyline, self).__init__()
        self.g_pool = g_pool
        self.order = .9
        self.prev_frame_idx = -1

    def update(self,frame,recent_pupil_positions,events):
        if self.prev_frame_idx != frame.index:
            pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]
            if pts:
                pts = np.array([pts],dtype=np.int32)
                cv2.polylines(frame.img, pts, isClosed=False, color=(0,255,0), thickness=1, lineType=cv2.cv.CV_AA)

            self.prev_frame_idx = frame.index


    def gl_display(self):
        pass
