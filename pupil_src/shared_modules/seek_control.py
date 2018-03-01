'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from bisect import bisect_left

from pyglui import ui
from plugin import System_Plugin_Base
import time

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
        self.trim_left = 0
        self.trim_right = len(self.g_pool.timestamps) - 1
        self.drag_mode = False
        self.was_playing = True
        self.start_time = 0.
        self.start_ts = 0.
        g_pool.capture.play = False

    def init_ui(self):
        self.seek_bar = ui.Seek_Bar(self, self.g_pool.timestamps[0],
                                    self.g_pool.timestamps[-1], self.on_seek,
                                    self.g_pool.user_timelines)
        self.g_pool.timelines.insert(0, self.seek_bar)

    def deinit_ui(self):
        self.g_pool.timelines.remove(self.seek_bar)
        self.seek_bar = None

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return

        if frame.timestamp == self.trim_left_ts or frame.timestamp == self.trim_right_ts:
            self.g_pool.capture.play = False

    def on_seek(self, seeking):
        if seeking:
            self.was_playing = self.g_pool.capture.play
            self.g_pool.capture.play = False
            self.start_time = 0.
            self.start_ts = 0.
        else:
            self.g_pool.capture.play = self.was_playing
            if self.g_pool.capture.play:
                self.start_time = time.monotonic()
                self.start_ts = self.current_ts

    @property
    def play(self):
        return self.g_pool.capture.play

    @play.setter
    def play(self, new_state):
        if new_state and self.current_ts == self.trim_right_ts:
            self.g_pool.capture.seek_to_frame(self.trim_left)
            self.g_pool.new_seek = True

        # Sometimes there is less video frames than timestamps. The rewind
        # logic needs to catch these cases but work for recordings with less
        # than 10 frames
        elif new_state and self.current_ts >= self.g_pool.timestamps[-10:][0]:
            self.g_pool.capture.seek_to_frame(0)
            self.g_pool.new_seek = True
            logger.warning("End of video - restart at beginning.")
        else:
            self.g_pool.capture.play = new_state
        if new_state:
            self.start_time = time.monotonic()
            self.start_ts = self.current_ts
        else:
            self.start_time = self.start_ts = 0.

        self.g_pool.capture.play = new_state

    @property
    def current_playback_time(self):
        if self.g_pool.capture.play:
            return (time.monotonic() - self.start_time) * self.playback_speed + self.start_ts
        else:
            return 0.

    @property
    def trim_left_ts(self):
        return self.g_pool.timestamps[self.trim_left]

    @trim_left_ts.setter
    def trim_left_ts(self, val):
        self.trim_left = bisect_left(self.g_pool.timestamps, val, hi=self.trim_right-1)

    @property
    def trim_right_ts(self):
        return self.g_pool.timestamps[self.trim_right]

    @trim_right_ts.setter
    def trim_right_ts(self, val):
        # left + 1 <= right <= frame_count -1
        self.trim_right = bisect_left(self.g_pool.timestamps, val, lo=self.trim_left+1)
        self.trim_right = min(self.trim_right, len(self.g_pool.timestamps) - 1)

    @property
    def current_ts(self):
        return self.g_pool.timestamps[self.g_pool.capture.get_frame_index()]

    @current_ts.setter
    def current_ts(self, val):
        if self.current_ts != val:
            try:
                val_idx = bisect_left(self.g_pool.timestamps, val)
                self.g_pool.capture.seek_to_frame(val_idx)
            except:
                import traceback as tb
                logger.error(tb.format_exc())
            self.g_pool.new_seek = True

    @property
    def forwards(self):
        pass

    @forwards.setter
    def forwards(self, x):
        if self.g_pool.capture.play:
            # playback mode, increase playback speed
            speeds = self.g_pool.capture.allowed_speeds
            old_idx = speeds.index(self.g_pool.capture.playback_speed)
            new_idx = min(len(speeds) - 1, old_idx + 1)
            self.g_pool.capture.playback_speed = speeds[new_idx]
            self.start_time = time.monotonic()
            self.start_ts = self.current_ts
        else:
            # frame-by-frame mode, seek one frame forward
            self.g_pool.new_seek = True

    @property
    def backwards(self):
        pass

    @backwards.setter
    def backwards(self, x):
        if self.g_pool.capture.play:
            # playback mode, decrease playback speed
            speeds = self.g_pool.capture.allowed_speeds
            old_idx = speeds.index(self.g_pool.capture.playback_speed)
            new_idx = max(0, old_idx - 1)
            self.g_pool.capture.playback_speed = speeds[new_idx]
            self.start_time = time.monotonic()
            self.start_ts = self.current_ts
        else:
            # frame-by-frame mode, seek one frame backwards
            self.g_pool.capture.seek_to_prev_frame()
            self.g_pool.new_seek = True

    @property
    def playback_speed(self):
        return self.g_pool.capture.playback_speed if self.g_pool.capture.play else 0.

    def set_trim_range(self, mark_range):
        self.trim_left, self.trim_right = mark_range

    def get_rel_time_trim_range_string(self):
        time_fmt = ''
        min_ts = self.g_pool.timestamps[0]
        for ts in (self.trim_left_ts, self.trim_right_ts):
            ts -= min_ts
            minutes = ts // 60
            seconds = ts - (minutes * 60.)
            micro_seconds_e1 = int((seconds - int(seconds)) * 1e3)
            time_fmt += '{:02.0f}:{:02d}.{:03d} - '.format(abs(minutes), int(seconds), micro_seconds_e1)
        return time_fmt[:-3]

    def set_rel_time_trim_range_string(self, range_str):
        try:
            range_list = range_str.split('-')
            assert len(range_list) == 2

            def convert_to_ts(time_str):
                time_list = time_str.split(':')
                assert len(time_list) == 2
                minutes, seconds = map(float, time_list)
                return minutes * 60 + seconds + self.g_pool.timestamps[0]

            left, right = map(convert_to_ts, range_list)
            assert left < right
            # setting right trim first ensures previous assertion even if
            # g_pool.timestamps[-1] < left < right. The possibility is
            # desirable, because it enables to set the range to the maximum
            # value without knowing the actual maximum value.
            self.trim_right_ts = right
            self.trim_left_ts = left

        except (AssertionError, ValueError):
            logger.warning('Invalid time range entered')

    def get_abs_time_trim_range_string(self):
        return '{} - {}'.format(self.g_pool.timestamps[self.trim_left],
                                self.g_pool.timestamps[self.trim_right])

    def get_frame_index_trim_range_string(self):
        return '{} - {}'.format(self.trim_left, self.trim_right)

    def set_frame_index_trim_range_string(self, range_str):
        try:
            range_list = range_str.split('-')
            assert len(range_list) == 2

            left, right = map(int, range_list)
            assert left < right

            self.trim_right = right
            self.trim_left = left

        except AssertionError:
            logger.warning('Invalid frame index range entered')

    def get_folder_name_from_trims(self):
        time_fmt = ''
        min_ts = self.g_pool.timestamps[0]
        for ts in (self.trim_left_ts, self.trim_right_ts):
            ts -= min_ts
            minutes = ts // 60
            seconds = ts - (minutes * 60.)
            micro_seconds_e1 = int((seconds - int(seconds)) * 1e3)
            time_fmt += '{:02.0f}_{:02d}_{:03d}-'.format(abs(minutes), int(seconds), micro_seconds_e1)
        return time_fmt[:-1]
