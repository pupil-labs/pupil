'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from pyglui import ui
from plugin import System_Plugin_Base

import logging
logger = logging.getLogger(__name__)


class Seek_Control(System_Plugin_Base):
    """docstring for Seek_Control
    seek bar displays a bar at the bottom of the screen when you hover close to it.
    it will show the current positon and allow you to drag to any postion in the video file.
    """
    order = 0.01

    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.seek_control = self
        self.current_frame_index = self.g_pool.capture.get_frame_index()
        self.frame_count = self.g_pool.capture.get_frame_count()
        self._trim_left = 0
        self._trim_right = self.frame_count - 1

        self.drag_mode = False
        self.was_playing = True

    def init_ui(self):
        self.seek_bar = ui.Seek_Bar(self, self.frame_count - 1, self.on_seek, self.g_pool.user_timelines)
        self.g_pool.timelines.append(self.seek_bar)

    def deinit_ui(self):
        self.g_pool.timelines.remove(self.seek_bar)
        self.seek_bar = None

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return

        if frame.index == self.trim_left or frame.index == self.trim_right:
            self.g_pool.capture.play = False

    def on_seek(self, seeking):
        if seeking:
            self.was_playing = self.g_pool.capture.play
            self.g_pool.capture.play = False
        else:
            try:
                self.g_pool.capture.seek_to_frame(self.current_index)
            except:
                pass
            self.g_pool.new_seek = True
            self.g_pool.capture.play = self.was_playing

    @property
    def play(self):
        return self.g_pool.capture.play

    @play.setter
    def play(self, new_state):
        if new_state and self.current_index >= self.frame_count - 1:
            self.g_pool.capture.seek_to_frame(0)  # avoid pause set by hitting trimmark pause.
            logger.warning("End of video - restart at beginning.")
        self.g_pool.capture.play = new_state

    @property
    def trim_left(self):
        return self._trim_left

    @trim_left.setter
    def trim_left(self, val):
        if val != self._trim_left:
            # 0 <= left <= right + 1
            self._trim_left = max(0, min(val, self.trim_right - 1))

    @property
    def trim_right(self):
        return self._trim_right

    @trim_right.setter
    def trim_right(self, val):
        if val != self._trim_right:
            # left + 1 <= right <= frame_count -1
            self._trim_right = max(self.trim_left + 1, min(val, self.frame_count - 1))

    @property
    def current_index(self):
        return self.g_pool.capture.get_frame_index()

    @current_index.setter
    def current_index(self, val):
        if self.current_index != val:
            try:
                # logger.info('seeking to {} form {}'.format(seek_pos,self.current_frame_index))
                self.g_pool.capture.seek_to_frame(val)
            except:
                pass
            self.g_pool.new_seek = True

    def set_trim_range(self, mark_range):
        self.trim_left, self.trim_right = mark_range

    def get_trim_range_string(self):
        return '{} - {}'.format(self.trim_left, self.trim_right)

    def set_trim_range_string(self, str):
        try:
            in_m, out_m = str.split('-')
            in_m = int(in_m)
            out_m = int(out_m)
            self.trim_left = in_m
            self.trim_right = out_m
        except:
            logger.warning("Setting Trimmarks via string failed.")
