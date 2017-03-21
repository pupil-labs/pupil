'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
from pyglui.cygl.utils import draw_points_norm,RGBA
from pyglui import ui


class Display_Recent_Gaze(Plugin):
    """
    DisplayGaze shows the three most
    recent gaze position on the screen
    """

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []

    def recent_events(self,events):
        for pt in events.get('gaze_positions',[]):
            self.pupil_display_list.append((pt['norm_pos'] , pt['confidence']))
        self.pupil_display_list[:-3] = []


    def gl_display(self):
        for pt,a in self.pupil_display_list:
            #This could be faster if there would be a method to also add multiple colors per point
            draw_points_norm([pt],
                        size=35,
                        color=RGBA(1.,.2,.4,a))

    def get_init_dict(self):
        return {}
