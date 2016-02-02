'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np

from pyglui import ui
from methods import denormalize
import logging
logger = logging.getLogger(__name__)

class Vis_Light_Points(Plugin):
    """docstring
    show gaze dots at light dots on numpy.

    """
    uniqueness = "not_unique"

    def __init__(self, g_pool,falloff = 20):
        super(Vis_Light_Points, self).__init__(g_pool)
        self.order = .8
        self.menu = None

        self.falloff = falloff

    def update(self,frame,events):
        falloff = self.falloff

        img = frame.img
        screen_gaze = [denormalize(g['norm_pos'],self.g_pool.capture.frame_size,flip_y=True) for g in events.get('gaze_positions',[])]

        overlay = np.ones(img.shape[:-1],dtype=img.dtype)

        # draw recent gaze postions as black dots on an overlay image.
        for gaze_point in screen_gaze:
            try:
                overlay[int(gaze_point[1]),int(gaze_point[0])] = 0
            except:
                pass

        out = cv2.distanceTransform(overlay,cv2.cv.CV_DIST_L2, 5)

        # fix for opencv binding inconsitency
        if type(out)==tuple:
            out = out[0]

        overlay =  1/(out/falloff+1)

        img *= cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Light Points')
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Slider('falloff',self,min=1,step=1,max=1000))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'falloff': self.falloff}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
