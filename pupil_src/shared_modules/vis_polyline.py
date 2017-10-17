'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import Visualizer_Plugin_Base
import numpy as np

import cv2

from pyglui import ui
from methods import denormalize


class Vis_Polyline(Visualizer_Plugin_Base):
    uniqueness = "not_unique"
    icon_chr = chr(0xe922)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool,color=(1.0,0.0,0.4,1.0),thickness=2):
        super().__init__(g_pool)
        self.order = .9
        self.menu = None

        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]
        self.thickness = thickness

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        pts = [denormalize(pt['norm_pos'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events.get('gaze_positions',[]) if pt['confidence']>=self.g_pool.min_data_confidence]
        bgra = (self.b*255,self.g*255,self.r*255,self.a*255)
        if pts:
            pts = np.array([pts],dtype=np.int32)
            cv2.polylines(frame.img, pts, isClosed=False, color=bgra, thickness=self.thickness, lineType=cv2.LINE_AA)

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Gaze Polyline'
        self.menu.append(ui.Slider('thickness',self,min=1,step=1,max=15,label='Line thickness'))

        color_menu = ui.Growing_Menu('Color')
        color_menu.collapsed = True
        color_menu.append(ui.Info_Text('Set RGB color component values.'))
        color_menu.append(ui.Slider('r',self,min=0.0,step=0.05,max=1.0,label='Red'))
        color_menu.append(ui.Slider('g',self,min=0.0,step=0.05,max=1.0,label='Green'))
        color_menu.append(ui.Slider('b',self,min=0.0,step=0.05,max=1.0,label='Blue'))
        self.menu.append(color_menu)

    def deinit_ui(self):
        self.remove_menu()

    def get_init_dict(self):
        return {'color':(self.r, self.g, self.b, self.a),'thickness':self.thickness}
