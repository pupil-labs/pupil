'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np
from pyglui import ui
from methods import denormalize,normalize
import logging
logger = logging.getLogger(__name__)

class Scan_Path(Plugin):
    """docstring
    using this plugin will extend the recent_pupil_positions by x extra dots from previous frames.
    lock recent gaze points onto pixels.
    """

    def __init__(self, g_pool,timeframe=.5,menu_conf={'pos':(10,390),'size':(300,70),'collapsed':False}):
        super(Scan_Path, self).__init__(g_pool)
        #let the plugin work after most other plugins.
        self.order = .6

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf
        #user settings
        self.timeframe = timeframe

        #algorithm working data
        self.prev_frame_idx = -1
        self.past_pupil_positions = []
        self.prev_gray = None


    def update(self,frame,events):
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
        if self.past_pupil_positions and succeeding_frame:
            past_screen_gaze = np.array([denormalize(ng['norm_gaze'] ,img_shape,flip_y=True) for ng in self.past_pupil_positions],dtype=np.float32)
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_img,past_screen_gaze,minEigThreshold=0.005,**lk_params)
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
        else:
            # we must be seeking, do not try to do optical flow, or pausing: see below.
            pass

        if same_frame:
            # paused
            # re-use last result
            events['pupil_positions'][:] = self.past_pupil_positions[:]
        else:
            # trim gaze that is too old
            if events['pupil_positions']:
                now = events['pupil_positions'][0]['timestamp']
                cutoff = now-self.timeframe
                updated_past_gaze = [g for g in updated_past_gaze if g['timestamp']>cutoff]

            #inject the scan path gaze points into recent_pupil_positions
            events['pupil_positions'][:] = updated_past_gaze + events['pupil_positions']
            events['pupil_positions'].sort(key=lambda x: x['timestamp']) #this may be redundant...

        #update info for next frame.
        self.prev_gray = gray_img
        self.prev_frame_idx = frame.index
        # copy the data/contents of recent_pupil_positions don't make a reference
        self.past_pupil_positions = events['pupil_positions'][:]

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Scan Path')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Slider('timeframe',self,min=0,step=0.1,max=5,label="duration in sec"))
        self.menu.append(ui.Button('remove',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'timeframe':self.timeframe, 'menu_conf':self.menu.configuration}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

