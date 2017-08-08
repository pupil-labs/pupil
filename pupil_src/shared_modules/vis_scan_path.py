'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Visualizer_Plugin_Base
import numpy as np
from pyglui import ui
from methods import denormalize,normalize
import logging
logger = logging.getLogger(__name__)


class Vis_Scan_Path(Visualizer_Plugin_Base):
    """docstring
    using this plugin will extend the recent_gaze_positions by x extra dots from previous frames.
    lock recent gaze points onto pixels.
    """

    def __init__(self, g_pool,timeframe=.5):
        super().__init__(g_pool)
        #let the plugin work after most other plugins.
        self.order = .6
        self.menu = None

        #user settings
        self.timeframe = timeframe

        #algorithm working data
        self.prev_frame_idx = -1
        self.past_gaze_positions = []
        self.prev_gray = None


    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        succeeding_frame = frame.index-self.prev_frame_idx == 1
        same_frame = frame.index == self.prev_frame_idx
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #vars for calcOpticalFlowPyrLK
        lk_params = dict( winSize  = (90, 90),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        updated_past_gaze = []

        #lets update past gaze using optical flow: this is like sticking the gaze points onto the pixels of the img.
        if self.past_gaze_positions and succeeding_frame:
            past_screen_gaze = np.array([denormalize(ng['norm_pos'] ,img_shape,flip_y=True) for ng in self.past_gaze_positions],dtype=np.float32)
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray,gray_img,past_screen_gaze,None,minEigThreshold=0.005,**lk_params)
            for gaze,new_gaze_pt,s,e in zip(self.past_gaze_positions,new_pts,status,err):
                if s:
                    # print "norm,updated",gaze['norm_gaze'], normalize(new_gaze_pt,img_shape[:-1],flip_y=True)
                    gaze['norm_pos'] = normalize(new_gaze_pt,img_shape,flip_y=True)
                    updated_past_gaze.append(gaze)
                    # logger.debug("updated gaze")

                else:
                    # logger.debug("dropping gaze")
                    # Since we will replace self.past_gaze_positions later,
                    # not appedning tu updated_past_gaze is like deliting this data point.
                    pass
        else:
            # we must be seeking, do not try to do optical flow, or pausing: see below.
            pass

        if same_frame:
            # paused
            # re-use last result
            events['gaze_positions'][:] = self.past_gaze_positions[:]
        else:
            # trim gaze that is too old
            if events['gaze_positions']:
                now = events['gaze_positions'][0]['timestamp']
                cutoff = now-self.timeframe
                updated_past_gaze = [g for g in updated_past_gaze if g['timestamp']>cutoff]

            #inject the scan path gaze points into recent_gaze_positions
            events['gaze_positions'][:] = updated_past_gaze + events['gaze_positions']
            events['gaze_positions'].sort(key=lambda x: x['timestamp']) #this may be redundant...

        #update info for next frame.
        self.prev_gray = gray_img
        self.prev_frame_idx = frame.index
        self.past_gaze_positions = events['gaze_positions']

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Scan Path')
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Slider('timeframe',self,min=0,step=0.1,max=5,label="duration in sec"))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'timeframe':self.timeframe}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
