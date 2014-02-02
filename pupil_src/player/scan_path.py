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
from methods import denormalize,normalize
import logging
logger = logging.getLogger(__name__)

class Scan_Path(Plugin):
    """docstring
    using this plugin will extend the recent_pupil_positions by x extra dots from previous frames.
    lock recent gaze points onto pixels.
    """

    def __init__(self, g_pool):
        super(Scan_Path, self).__init__()
        self.g_pool = g_pool

        #let the plugin work after most other plugins.
        self.order = .6


        #user settings
        self.scan_path_timeframe = 3.

        #algorithm working data
        self.prev_frame_idx = -1
        self.past_pupil_positions = []
        self.prev_gray = None


    def update(self,frame,recent_pupil_positions,events):
        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        succeeding_frame = frame.index-self.prev_frame_idx == 1
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #vars for calcOpticalFlowPyrLK
        lk_params = dict( winSize  = (90, 90),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


        updated_past_gaze = []

        #lets update past gaze using optical flow: this is like sticking the gaze points onto the pixels of the img.
        if self.past_pupil_positions and succeeding_frame:
            past_screen_gaze = np.array([denormalize(ng['norm_gaze'] ,img_shape,flip_y=True) for ng in self.past_pupil_positions],dtype=np.float32)
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_img,past_screen_gaze,minEigThreshold=0.01,**lk_params)

            for gaze,new_gaze_pt,s,e in zip(self.past_pupil_positions,new_pts,status,err):
                if s:
                    # print "norm,updated",gaze['norm_gaze'], normalize(new_gaze_pt,img_shape[:-1],flip_y=True)
                    gaze['norm_gaze'] = normalize(new_gaze_pt,img_shape,flip_y=True)
                    updated_past_gaze.append(gaze)
                    # logger.debug("updated gaze")

                else:
                    # logger.debug("dropping gaze")
                    # Since we will replace self.past_pupil_positions later,
                    # not appedning tu updated_past_gaze is like deliting this data point.
                    pass


        # print "new_gaze", len(recent_pupil_positions)
        # print "from before", len(updated_past_gaze)

        # trim of gaze that is too old
        if recent_pupil_positions:
            now = recent_pupil_positions[0]['timestamp']
            cutof = now-self.scan_path_timeframe
            updated_past_gaze = [g for g in updated_past_gaze if g['timestamp']>cutof]


        #inject the scan path gaze points into recent_pupil_positions
        recent_pupil_positions[:] = updated_past_gaze + recent_pupil_positions
        recent_pupil_positions.sort(key=lambda x: x['timestamp']) #this may be redundant...


        #update info for next frame.
        self.prev_gray = gray_img
        self.prev_frame_idx = frame.index
        self.past_pupil_positions = recent_pupil_positions


    def gl_display(self):
        pass


