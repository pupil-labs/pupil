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
import numpy as np
import logging
logger = logging.getLogger(__name__)


class Blink_Detection(Plugin):
    """
    This plugin implements a blink detection algorithm, based on sudden drops in the
    pupil detection confidence.
    """
    order = .8

    def __init__(self, g_pool, history_length=0.2, onset_confidence_threshold=0.5, offset_confidence_threshold=0.5):
        super(Blink_Detection, self).__init__(g_pool)
        self.history_length = history_length  # unit: seconds
        self.onset_confidence_threshold = onset_confidence_threshold
        self.offset_confidence_threshold = offset_confidence_threshold

        self.history = deque()
        self.menu = None

    def init_gui(self):
        self.menu = ui.Growing_Menu('Blink Detector')
        self.g_pool.sidebar.append(self.menu)
        self.menu.append(ui.Button('Close', self.close))
        self.menu.append(ui.Info_Text('This plugin detects blink on- and offsets based on confidence drops.'))
        self.menu.append(ui.Slider('history_length', self,
                                   label='Filter length [seconds]',
                                   min=0.1, max=.5, step=.05))
        self.menu.append(ui.Slider('onset_confidence_threshold', self,
                                   label='Onset confidence threshold',
                                   min=0., max=1., step=.05))
        self.menu.append(ui.Slider('offset_confidence_threshold', self,
                                   label='Offset confidence threshold',
                                   min=0., max=1., step=.05))

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
        blink_filter = np.ones(filter_size) / filter_size
        blink_filter[filter_size // 2:] *= -1

        # The theoretical response maximum is +-0.5
        # Response of +-0.45 seems sufficient for a confidence of 1.
        filter_response = activity @ blink_filter / 0.45

        if -self.offset_confidence_threshold <= filter_response <= self.onset_confidence_threshold:
            return  # response cannot be classified as blink onset or offset

        if filter_response > self.onset_confidence_threshold:
            blink_type = 'onset'
        else:
            blink_type = 'offset'

        confidence = min(abs(filter_response), 1.)  # clamp conf. value at 1.
        logger.debug('Blink {} detected with confidence {:0.3f}'.format(blink_type, confidence))
        # Add info to events
        blink_entry = {
            'topic': 'blink',
            'type': blink_type,
            'confidence': confidence,
            'base_data': list(self.history),
            'timestamp': self.history[len(self.history)//2]['timestamp'],
        }
        events['blinks'].append(blink_entry)

    def get_init_dict(self):
        return {'history_length': self.history_length,
                'onset_confidence_threshold': self.onset_confidence_threshold,
                'offset_confidence_threshold': self.offset_confidence_threshold}
