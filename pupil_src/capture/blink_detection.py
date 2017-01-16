'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
from pyglui.cygl.utils import draw_points_norm,RGBA
from collections import deque
from itertools import islice
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)


class Blink_Detection(Plugin):
    """
    DisplayGaze shows the three most
    recent gaze position on the screen
    """

    def __init__(self, g_pool):
        super(Blink_Detection, self).__init__(g_pool)
        self.order = .8
        self.history_length_per_fps = 0.2

        # The maximum length of the history needs to be set a priori. If we are assuming a maximum framerat of 120 FPS
        # and a genorous maximum on-set duratio of a blink of 0.5 seconds, 60 frames of history should always be enough
        self.confidence_histories = (deque(maxlen=60), deque(maxlen=60))
        self.timestamp_histories = (deque(maxlen=60), deque(maxlen=60))
        self.eyes_are_alive = g_pool.eyes_are_alive

        self.is_blink = False

    def update(self, frame=None, events={}):
        # backwards compatibility
        self.recent_events(events)

    def recent_events(self, events={}):

        # Process all pupil_positions events
        for pt in events.get('pupil_positions',[]):
            # Update history
            self.confidence_histories[pt['id']].appendleft(pt['confidence'])
            self.timestamp_histories[pt['id']].appendleft(pt['timestamp'])

            # Wait for at least 5 frames of history to compute the current framerate with
            if len(self.timestamp_histories[pt['id']]) < 60:
                continue
            else:
                fps = 60.0 / (self.timestamp_histories[pt['id']][0] -  self.timestamp_histories[pt['id']][59])
                # fps = 120
                self.history_length = int(self.history_length_per_fps * fps)

            # Build filter based on current history length
            filter = np.asarray([-1 for i in range(int(math.floor(self.history_length / 2.0)))] + [1 for i in range(
                int(math.ceil(self.history_length / 2.0)))])
            filter = np.ones(self.hist_len)
            filter[self.hist_len // 2:] = -1

            # Compute activations if history is sufficient
            if self.eyes_are_alive[0].value:
                if len(self.confidence_histories[0]) >= self.history_length:
                    slice = np.asarray(deque(islice(self.confidence_histories[0], 0, self.history_length)))
                    # Normalize deviations in the overall magnitude and legth of the used filter
                    slice = (slice - slice.min()) / (slice.max() - slice.min())
                    act0 = np.dot(slice, filter) / self.history_length
                else:
                    continue

            if self.eyes_are_alive[1].value:
                if len(self.confidence_histories[1]) >= self.history_length:
                    slice = np.asarray(deque(islice(self.confidence_histories[1], 0, self.history_length)))
                    # Normalize deviations in the overall magnitude and legth of the used filter
                    slice = (slice - slice.min()) / (slice.max() - slice.min())
                    act1 = np.dot(slice, filter) / self.history_length
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

            print str(self.is_blink) + " " + str(act)

            # Add info to events
            blink_entry = {
                'topic': 'blink',
                'activation': act,
                'timestamp': self.timestamp_histories[pt['id']][int(math.ceil(self.history_length / 2.0))],
                'is_blink': self.is_blink
            }

            if not 'blinks' in events:
                events['blinks'] = []
            events['blinks'].append(blink_entry)


    def get_init_dict(self):
        return {}
