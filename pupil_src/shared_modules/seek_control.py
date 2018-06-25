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
    available_speeds = [.25, .5, 1., 1.5, 2., 4.]

    def __init__(self, g_pool, playback_speed=1.):
        super().__init__(g_pool)
        g_pool.seek_control = self
        self._playback_speed = playback_speed
        self.trim_left = 0
        self.trim_right = len(self.g_pool.timestamps) - 1
        self.was_playing = True
        self.was_seeking = False
        self.start_time = 0.
        self.start_ts = self.g_pool.timestamps[0]
        self.time_slew = 0.
        g_pool.capture.play = False
        self._recent_playback_time = self.current_playback_time

    def init_ui(self):
        self.seek_bar = ui.Seek_Bar(sync_ctx=self,
                                    min_ts=self.g_pool.timestamps[0],
                                    max_ts=self.g_pool.timestamps[-1],
                                    recent_idx_ts_getter=self.g_pool.capture.get_frame_index_ts,
                                    playback_time_setter=self.set_playback_time,
                                    seeking_cb=self.on_seek,
                                    handle_start_reference=self.g_pool.user_timelines)
        self.g_pool.timelines.insert(0, self.seek_bar)

    def deinit_ui(self):
        self.g_pool.timelines.remove(self.seek_bar)
        self.seek_bar = None

    def recent_events(self, events):
        pbt = events['frame'].timestamp
        if self.play and self._recent_playback_time < self.trim_left_ts <= pbt:
            self._recent_playback_time = self.trim_left_ts
            self.play = False
            self.start_ts = self.trim_left_ts
        elif self.play and self._recent_playback_time < self.trim_right_ts <= pbt:
            self._recent_playback_time = self.trim_right_ts
            self.play = False
            self.start_ts = self.trim_right_ts
        else:
            self._recent_playback_time = pbt

    def on_seek(self, seeking):
        if seeking:
            self.was_seeking = True
            self.was_playing = self.play
            self.play = False
            self.notify_all({'subject': 'seek_control.was_seeking'})
        elif self.was_playing:
            self.play = True

    @property
    def play(self):
        return self.g_pool.capture.play

    @play.setter
    def play(self, new_state):
        if new_state and self._recent_playback_time == self.trim_right_ts:
            self.start_ts = self.trim_left_ts
            new_state = False  # Do not auto-play on rewind

        # Sometimes there is less video frames than timestamps. The rewind
        # logic needs to catch these cases but work for recordings with less
        # than 10 frames
        elif new_state and self._recent_playback_time >= self.g_pool.timestamps[-10:][0]:
            self.start_ts = self.g_pool.timestamps[0]
            logger.warning("End of video - restart at beginning.")
            new_state = False  # Do not auto-play on rewind

        elif not new_state:
            self.start_ts = self.g_pool.capture.get_frame_index_ts()[1]

        self.start_time = time.monotonic()
        self.g_pool.capture.play = new_state
        self.time_slew = 0

    @property
    def current_playback_time(self):
        playback_time = self.start_ts - self.time_slew

        if self.g_pool.capture.play:
            playback_time += (time.monotonic() - self.start_time) * self._playback_speed
        return playback_time

    def on_notify(self, notification):
        if notification['subject'] == 'seek_control.should_seek':
            if 'index' in notification:
                self.set_playback_time_idx(notification['index'])
            elif 'timestamp' in notification:
                self.set_playback_time(notification['timestamp'])

    def set_playback_time(self, val):
        '''Callback used by seek bar on user input'''
        idx = self.ts_idx_from_playback_time(val)
        self.set_playback_time_idx(idx)

    def set_playback_time_idx(self, idx):
        '''Callback used by plugins to request seek'''
        self.start_ts = self.g_pool.timestamps[idx]
        self.was_seeking = True
        self.notify_all({'subject': 'seek_control.was_seeking'})

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

    def ts_idx_from_playback_time(self, playback_time):
        all_ts = self.g_pool.timestamps
        index = bisect_left(all_ts, playback_time)
        if index == len(all_ts):
            index = len(all_ts) - 1
        return index

    @property
    def forwards(self):
        pass

    @forwards.setter
    def forwards(self, x):
        recent_idx, recent_ts = self.g_pool.capture.get_frame_index_ts()
        if self.g_pool.capture.play:
            self.start_ts = recent_ts
            self.start_time = time.monotonic()
            # playback mode, increase playback speed
            old_idx = self.available_speeds.index(self._playback_speed)
            new_idx = min(len(self.available_speeds) - 1, old_idx + 1)
            self._playback_speed = self.available_speeds[new_idx]
            self.time_slew = 0
        else:
            # frame-by-frame mode, seek one frame forward
            ts_idx = recent_idx
            ts_idx = min(ts_idx + 1, len(self.g_pool.timestamps) - 1)
            self.start_ts = self.g_pool.timestamps[ts_idx]

    @property
    def backwards(self):
        pass

    @backwards.setter
    def backwards(self, x):
        recent_idx, recent_ts = self.g_pool.capture.get_frame_index_ts()
        if self.g_pool.capture.play:
            self.start_ts = recent_ts
            self.start_time = time.monotonic()
            # playback mode, decrease playback speed
            old_idx = self.available_speeds.index(self._playback_speed)
            new_idx = max(0, old_idx - 1)
            self._playback_speed = self.available_speeds[new_idx]
        else:
            # frame-by-frame mode, seek one frame forward
            ts_idx = recent_idx
            ts_idx = max(0, ts_idx - 1)
            self.start_ts = self.g_pool.timestamps[ts_idx]

    @property
    def playback_speed(self):
        return self._playback_speed if self.play else 0.

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

    def wait(self, ts):
        if self.play and not self.was_seeking:
            playback_now = self.current_playback_time
            time_diff = (ts - playback_now) / self._playback_speed
            if time_diff > .005:
                time.sleep(time_diff)
        else:
            time.sleep(1 / 60)
            self.was_seeking = False

    def get_init_dict(self):
        return {'playback_speed': self._playback_speed}
