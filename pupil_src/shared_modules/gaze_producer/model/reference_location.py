"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class ReferenceLocation:
    version = 1

    def __init__(self, screen_pos, frame_index, timestamp):
        self.screen_pos = screen_pos  # 2D tuple
        self.frame_index = frame_index
        self.timestamp = timestamp

    @property
    def screen_x(self):
        return self.screen_pos[0]

    @property
    def screen_y(self):
        return self.screen_pos[1]
