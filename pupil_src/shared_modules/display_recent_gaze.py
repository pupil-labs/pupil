"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

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
        self.pupil_display_list = []

    def recent_events(self, events):
        for pt in events.get("gaze", []):
            self.pupil_display_list.append((pt["norm_pos"], pt["confidence"] * 0.8))
        self.pupil_display_list[:-15] = []

    def gl_display(self):

        size = 60
        for pt, a in self.pupil_display_list:

            draw_circle(
                center_position=(pt[0] * 1080, (1 - pt[1]) * 1080),
                radius=size,
                stroke_width=40,
                color=RGBA(0.1, 0.4, 0.8, 0.3),
                sharpness=0.2,
            )
            size += 1.3

        # for pt, a in self.pupil_display_list:
        #     # This could be faster if there would be a method to also add multiple colors per point
        #     draw_points_norm([pt], size=35, color=RGBA(1.0, 0.2, 0.4, a))

    def get_init_dict(self):
        return {}
