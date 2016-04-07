'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
from pyglui.cygl.utils import draw_points_norm,RGBA
from pyglui import ui

class Smoothing_Filter(object):
    def __init__(self):
        super(Smoothing_Filter, self).__init__()
        self.prev = None
        self.prev_ts = None
        self.smoother = 0.5
        self.cut_dist = 0.01


    def filter(self,vals,ts):
        self.prev = vals
        self.pref_ts = ts
        self.filter = self._filter
        return vals


    def _filter(self,vals,ts):
        result = []
        for v,ov in zip(vals,self.prev):
            if abs(ov-v)>self.cut_dist:
                self.prev = tuple(vals)
                return vals
            else:
                result.append(ov+self.smoother*(v-ov))
        self.prev = result
        return type(vals)(result)


class Display_Recent_Gaze(Plugin):
    """
    DisplayGaze shows the three most
    recent gaze position on the screen
    """

    def __init__(self, g_pool, filter_active=True):
        super(Display_Recent_Gaze, self).__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []
        self.filter_active = filter_active
        self.filter = Smoothing_Filter()



    def update(self,frame,events):
        if self.filter_active:
            for pt in events.get('gaze_positions',[]):
                self.pupil_display_list.append( (self.filter.filter(pt['norm_pos'],pt['timestamp']), pt['confidence'] ) )
        else:
            for pt in events.get('gaze_positions',[]):
                self.pupil_display_list.append((pt['norm_pos'] , pt['confidence']))
        self.pupil_display_list[:-3] = []

    def init_gui(self):
        self.filter_switch = ui.Switch('filter_active',self,label='Smooth gaze visualization')
        self.g_pool.sidebar[0].insert(-1,self.filter_switch)

    def deinit_gui(self):
        if self.filter_switch:
            self.g_pool.sidebar[0].remove(self.filter_switch)
            self.filter_switch = None

    def cleanup(self):
        self.deinit_gui()

    def gl_display(self):
        for pt,a in self.pupil_display_list:
            #This could be faster if there would be a method to also add multiple colors per point
            draw_points_norm([pt],
                        size=35,
                        color=RGBA(1.,.2,.4,a))

    def get_init_dict(self):
        return {'filter_active':True}
