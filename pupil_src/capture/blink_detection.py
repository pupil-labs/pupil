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
import numpy as np
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
        self.history_length = 21 # Should be odd
        self.confidence_histories = (deque(maxlen=self.history_length), deque(maxlen=self.history_length))
        self.timestamp_histories = (deque(maxlen=self.history_length), deque(maxlen=self.history_length))
        self.eyes_are_alive = g_pool.eyes_are_alive

        self.is_blink = False

    def update(self,frame,events):
        filter = np.asarray([1 for i in range((self.history_length-1) / 2 + 1)] + [-1 for i in range((self.history_length-1) / 2)])

        # Update history
        for pt in events.get('pupil_positions',[]):
            self.confidence_histories[pt['id']].append(pt['confidence'])
            self.timestamp_histories[pt['id']].append(pt['timestamp'])

            # Compute activations if history is sufficient
            if self.eyes_are_alive[0] and len(self.confidence_histories[0]) < self.history_length:
                continue
            else:
                act0 = np.dot(self.confidence_histories[0], filter)

            if self.eyes_are_alive[1] and len(self.confidence_histories[1]) < self.history_length:
                continue
            else:
                act1 = np.dot(self.confidence_histories[1], filter)

            if self.eyes_are_alive[0] and self.eyes_are_alive[1]:
                act = np.maximum(act0, act1)
            elif self.eyes_are_alive[0]:
                act = act0
            else:
                act = act1

            if not self.is_blink and act > 7:
                logger.error("Bink")
                self.is_blink = True
            if self.is_blink and act < -2:
                self.is_blink = False

            print str(self.is_blink) + " " + str(act)

            blink_entry = {
                'topic': 'blink',
                'activation': act,
                'timestamp': self.timestamp_histories[pt['id']][10],
                'is_blink': int(self.is_blink)
            }

            if not 'blinks' in events:
                events['blinks'] = []
            events['blinks'].append(blink_entry)





    def gl_display(self):
        if not self.is_blink:
            draw_points_norm([(0.5,0.5)],
                             size=35,
                             color=RGBA(1., .2, .4, 1.))


    def get_init_dict(self):
        return {}
