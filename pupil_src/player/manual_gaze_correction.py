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
from copy import deepcopy
from plugin import Analysis_Plugin_Base
import numpy as np
from methods import denormalize,normalize
from player_methods import correlate_data
from pyglui import ui
import logging
logger = logging.getLogger(__name__)


class Manual_Gaze_Correction(Analysis_Plugin_Base):
    """docstring
    correct gaze with manually set x and y offset
    """

    def __init__(self, g_pool,x_offset=0.,y_offset=0.):
        super().__init__(g_pool)
        #let the plugin work before most other plugins.
        self.order = .3
        self.menu = None
        self.untouched_gaze_positions_by_frame = correlate_data(deepcopy(g_pool.pupil_data["gaze_positions"]), g_pool.timestamps)
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)
        self._set_offset()

    def _set_offset(self):
        x,y = self.x_offset,self.y_offset
        for f in range(len(self.g_pool.gaze_positions_by_frame)):
            for i in range(len(self.g_pool.gaze_positions_by_frame[f])):
                gaze_pos = self.untouched_gaze_positions_by_frame[f][i]['norm_pos']
                gaze_pos = gaze_pos[0]+x, gaze_pos[1]+y
                self.g_pool.gaze_positions_by_frame[f][i]['norm_pos'] =  gaze_pos
        self.notify_all({'subject':'gaze_positions_changed','delay':3})

    def _set_offset_x(self,offset_x):
        self.x_offset = offset_x
        self._set_offset()

    def _set_offset_y(self,offset_y):
        self.y_offset = offset_y
        self._set_offset()

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Manual Gaze Correction')
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Info_Text('Move gaze horizontally and vertically. Screen width and height are one unit respectively.'))
        self.menu.append(ui.Slider('x_offset',self,min=-1,step=0.01,max=1,setter=self._set_offset_x))
        self.menu.append(ui.Slider('y_offset',self,min=-1,step=0.01,max=1,setter=self._set_offset_y))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'x_offset':self.x_offset,'y_offset':self.y_offset}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.g_pool.gaze_positions_by_frame = self.untouched_gaze_positions_by_frame
        self.notify_all({'subject':'gaze_positions_changed'})
        self.deinit_gui()
