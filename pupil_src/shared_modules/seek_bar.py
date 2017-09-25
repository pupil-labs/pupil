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


class Seek_Bar(System_Plugin_Base):
    """docstring for Seek_Bar
    seek bar displays a bar at the bottom of the screen when you hover close to it.
    it will show the current positon and allow you to drag to any postion in the video file.
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.current_frame_index = self.g_pool.capture.get_frame_index()
        self.frame_count = self.g_pool.capture.get_frame_count()

        self.drag_mode = False
        self.was_playing = True

    def init_ui(self):
        self.seek_bar = ui.Seek_Bar(self, self.frame_count, self.on_seek)
        self.g_pool.timelines.append(self.seek_bar)

    def deinit_ui(self):
        self.g_pool.timelines.remove(self.seek_bar)
        self.seek_bar = None

    def recent_events(self, events):
        pass

    def on_seek(self, seeking):
        """
        gets called when the user clicks in the window screen
        """
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
    def trim_left(self):
        return 0

    @trim_left.setter
    def trim_left(self, val):
        pass

    @property
    def trim_right(self):
        return self.frame_count

    @trim_right.setter
    def trim_right(self, val):
        pass

    @property
    def current_index(self):
        return self.g_pool.capture.get_frame_index()

    @current_index.setter
    def current_index(self, val):
        if self.seek_bar.dragging and self.current_index != val:
            try:
                # logger.info('seeking to {} form {}'.format(seek_pos,self.current_frame_index))
                self.g_pool.capture.seek_to_frame(val)
            except:
                pass
            self.g_pool.new_seek = True
