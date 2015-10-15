'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
from pyglui.cygl.utils import draw_points_norm,RGBA
from one_euro_filter import OneEuroFilter
from pyglui import ui

class Display_Recent_Gaze(Plugin):
    """
    DisplayGaze shows the three most
    recent gaze position on the screen
    """

    def __init__(self, g_pool, filter_active):
        super(Display_Recent_Gaze, self).__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []
        self.filter_active = filter_active
        
        config = {
            'freq': 30,        # Hz
            'mincutoff': 1, # lower values decrease jitter
            'beta': 5,    # higher values decrease lag
            'dcutoff': 1.0     # this one should be ok
        }
        self._filter_x = OneEuroFilter(**config)
        self._filter_y = OneEuroFilter(**config)
        
    def filter(self, norm_pos):
        if self.filter_active:
            return self._filter_x(norm_pos[0]), self._filter_y(norm_pos[1])
        else:
            return norm_pos

    def update(self,frame,events):
        for pt in events.get('gaze_positions',[]):
            self.pupil_display_list.append(self.filter(pt['norm_pos']))

        self.pupil_display_list[:-3] = []
        
    def init_gui(self):
        self.filter_switch = ui.Switch('filter_active',self,on_val=True,off_val=False,label='Smooth gaze visualization')
        self.g_pool.sidebar[0].insert(-1,self.filter_switch)
        
    def deinit_gui(self):
        if self.filter_switch:
            self.g_pool.sidebar[0].remove(self.filter_switch)
            self.filter_switch = None
            
    def cleanup(self):
        self.deinit_gui()

    def gl_display(self):
        draw_points_norm(self.pupil_display_list,
                        size=35,
                        color=RGBA(1.,.2,.4,.6))

    def get_init_dict(self):
        return {'filter_active':True}
