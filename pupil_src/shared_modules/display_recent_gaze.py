"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import deque

from plugin import System_Plugin_Base
from pyglui.cygl.utils import draw_points_norm, RGBA, draw_circle, draw_points


class Display_Recent_Gaze(System_Plugin_Base):
    """
    DisplayGaze shows the three most
    recent gaze position on the screen
    """

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = 0.8
        self.pupil_display_list = deque(maxlen=1)

    def recent_events(self, events):
        for pt in events.get("gaze", []):
            self.pupil_display_list.append(pt["norm_pos"])

    def gl_display(self):

        size = 70
        for pt in self.pupil_display_list:
            draw_circle(
                center_position=(pt[0] * 1080, (1 - pt[1]) * 1080),
                radius=size + 75,
                stroke_width=145,
                color=RGBA(0.0, 0.0, 0.0, 0.8),
                sharpness=0.15,
            )
            draw_circle(
                center_position=(pt[0] * 1080, (1 - pt[1]) * 1080),
                radius=size,
                stroke_width=10,
                color=RGBA(1, 1, 1, 0.6),
                sharpness=0.7,
            )
