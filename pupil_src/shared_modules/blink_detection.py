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
        self.history_length_per_fps = 0.2

        # The maximum length of the history needs to be set a priori. If we are assuming a maximum frame rate of 120 FPS
        # and a generous maximum onset duration of a blink of 0.5 seconds, 60 frames of history should always be enough
        self.confidence_histories = (deque(maxlen=60), deque(maxlen=60))
        self.timestamp_histories = (deque(maxlen=60), deque(maxlen=60))
        self.eyes_are_alive = g_pool.eyes_are_alive

        self.is_blink = False
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

        # Process all pupil_positions events
        for pt in events.get('pupil_positions', []):
            # Update history
            self.confidence_histories[pt['id']].appendleft(pt['confidence'])
            self.timestamp_histories[pt['id']].appendleft(pt['timestamp'])

            # Wait for at least 5 frames of history to compute the current frame rate with
            if len(self.timestamp_histories[pt['id']]) < 60:
                continue
            else:
                fps = 60.0 / (self.timestamp_histories[pt['id']][0] - self.timestamp_histories[pt['id']][59])
                # fps = 120
                self.history_length = int(self.history_length_per_fps * fps)

            # Build filter_ based on current history length
            filter_ = np.ones(self.history_length)
            filter_[self.history_length // 2:] = -1

            # Compute activations if history is sufficient
            if self.eyes_are_alive[0].value:
                if len(self.confidence_histories[0]) >= self.history_length:
                    slice = np.asarray(deque(islice(self.confidence_histories[0], 0, self.history_length)))
                    # Normalize deviations in the overall magnitude and length of the used filter_
                    if slice.max() > slice.min():
                        slice = (slice - slice.min()) / (slice.max() - slice.min())
                        act0 = np.dot(slice, filter_) / self.history_length
                    else:
                        act0 = 0.0
                else:
                    continue

            if self.eyes_are_alive[1].value:
                if len(self.confidence_histories[1]) >= self.history_length:
                    slice = np.asarray(deque(islice(self.confidence_histories[1], 0, self.history_length)))
                    # Normalize deviations in the overall magnitude and length of the used filter_
                    if slice.max() > slice.min():
                        slice = (slice - slice.min()) / (slice.max() - slice.min())
                        act1 = np.dot(slice, filter_) / self.history_length
                    else:
                        act1 = 0.0
                else:
                    continue

            # Combine activations if we are binocular
            if self.eyes_are_alive[0].value and self.eyes_are_alive[1].value:
                act = np.maximum(act0, act1)
            elif self.eyes_are_alive[0].value:
                act = act0
            elif self.eyes_are_alive[1].value:
                act = act1
            else:
                return

            # Judge if activation is sufficient for the on-set of a blink
            if not self.is_blink and act > 0.4:
                logger.error("Blink")
                self.is_blink = True

            # If we also want to measure the off-set of a blink we could do it like this
            if self.is_blink and act < -0.1:
                self.is_blink = False

            logger.debug('{} {}'.format(self.is_blink, act))

            # Add info to events
            blink_entry = {
                'topic': 'blink',
                'activation': act,
                'timestamp': self.timestamp_histories[pt['id']][int(math.ceil(self.history_length / 2.0))],
                'is_blink': self.is_blink
            }

            if 'blinks' not in events:
                events['blinks'] = []
            events['blinks'].append(blink_entry)

    def get_init_dict(self):
        return {}