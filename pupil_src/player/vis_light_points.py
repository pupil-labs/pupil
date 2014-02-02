'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np
import atb
from methods import denormalize
import logging
logger = logging.getLogger(__name__)

class Vis_Light_Points(Plugin):
    """docstring
    show gaze dots at light dots on numpy.

    """
    #let the plugin work after most other plugins.

    def __init__(self, g_pool):
        super(Vis_Light_Points, self).__init__()
        self.g_pool = g_pool

        self.order = .8

        self.prev_frame_idx = -1

    def update(self,frame,recent_pupil_positions,events):

        #since we edit the img inplace we should not do it in pause mode...
        if self.prev_frame_idx != frame.index:
            img = frame.img
            img_shape = img.shape[:-1][::-1]#width,height
            norm_gaze = [ng['norm_gaze'] for ng in recent_pupil_positions if ng['norm_gaze'] is not None]
            screen_gaze = [denormalize(ng,img_shape,flip_y=True) for ng in norm_gaze]


            overlay = np.ones(img.shape[:-1],dtype=img.dtype)

            # draw recent gaze postions as black dots on an overlay image.
            for gaze_point in screen_gaze:
                try:
                    overlay[int(gaze_point[1]),int(gaze_point[0])] = 0
                except:
                    pass

            out = cv2.distanceTransform(overlay,cv2.cv.CV_DIST_L2, 5)

            # fix for opencv binding incositency
            if type(out)==tuple:
                out = out[0]

            overlay =  1/(out/20+1)

            img *= cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)

            self.prev_frame_idx = frame.index



    def gl_display(self):
        pass


