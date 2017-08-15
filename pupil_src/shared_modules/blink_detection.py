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
from pyglui import ui
from collections import deque
from itertools import islice
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)


class Blink_Detection(Plugin):
    """
    This plugin implements a blink detection algorithm, based on sudden drops in the
    pupil detection confidence.
    """
    order = .8

    def __init__(self, g_pool):
        super(Blink_Detection, self).__init__(g_pool)
        self.history_length = 0.2  # unit: seconds

        # self.minimum_change = 0.7  # minimum difference between min and max confidence value in history
        self.minimum_onset_response = 0.3
        self.minimum_offset_response = 0.3

        self.history = deque()
        self.menu = None

    def init_gui(self):
        self.menu = ui.Growing_Menu('Blink Detector')
        self.g_pool.sidebar.append(self.menu)
        self.menu.append(ui.Info_Text('This plugin detects blinks based on binocular confidence drops.'))
        self.menu.append(ui.Button('Close', self.close))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def close(self):
        self.alive = False

    def cleanup(self):
        self.deinit_gui()

    def recent_events(self, events={}):
        events['blinks'] = []
        self.history.extend(events.get('pupil_positions', []))

        try:  # use newest gaze point to determine age threshold
            age_threshold = self.history[-1]['timestamp'] - self.history_length
            while self.history[1]['timestamp'] < age_threshold:
                self.history.popleft()  # remove outdated gaze points
        except IndexError:
            pass

        filter_size = len(self.history)
        if filter_size < 2 or self.history[-1]['timestamp'] - self.history[0]['timestamp'] < self.history_length:
            return

        activity = np.fromiter((pp['confidence'] for pp in self.history), dtype=float)
        # if activity.max() - activity.min() < self.minimum_change:
        #     return

        # Build blink_filter based on current history length
        blink_filter = np.ones(filter_size) / filter_size
        blink_filter[filter_size // 2:] *= -1

        # # normalize activity
        # activity -= activity.min()
        # activity /= activity.max()

        filter_response = activity @ blink_filter

        if -self.minimum_offset_response <= filter_response <= self.minimum_onset_response:
            return  # response cannot be classified as blink onset or offset

        if filter_response > self.minimum_onset_response:
            logger.warning('onset  {:0.3f}'.format(filter_response))
        else:
            logger.warning('offset {:0.3f}'.format(filter_response))

        # Add info to events
        blink_entry = {
            'topic': 'blink',
            'filter_response': filter_response,
            'base_data': list(self.history),
            'timestamp': self.history[len(self.history)//2]['timestamp'],
            'is_blink': bool(filter_response > self.minimum_onset_response)
        }
        events['blinks'].append(blink_entry)

    def get_init_dict(self):
        return {}
