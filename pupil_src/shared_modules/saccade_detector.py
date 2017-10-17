'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from math import sqrt, degrees
from collections import deque
from pyglui import ui
from plugin import Plugin

import logging
logger = logging.getLogger(__name__)


class Saccade_Detector(Plugin):
    icon_chr = chr(0xe8f5)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, dispersion_2d=5, dispersion_3d=3.0, max_time_delta=0.1, confidence_threshold=0.75):
        super().__init__(g_pool)
        self.history = deque(maxlen=2), deque(maxlen=2)
        self.max_time_delta = max_time_delta
        self.confidence_threshold = confidence_threshold
        self.dispersion_3d = dispersion_3d

    def recent_events(self, events):
        self.recent_saccades = []
        for p in events.get('pupil_positions', []):
            self.process_pupil_position(p)

        events['saccades'] = self.recent_saccades

    def process_pupil_position(self, datum):
        if datum['confidence'] < self.confidence_threshold:
            return

        eyeid = datum['id']
        self.history[eyeid].append(datum)

        # test if self.history[eyeid] fullfills basic saccade requirements
        t1 = self.history[eyeid][0]['timestamp']
        t2 = self.history[eyeid][-1]['timestamp']
        if len(self.history[eyeid]) < 2 or t2 - t1 > self.max_time_delta:
            return

        # we only have a maximum of two data points in the history
        use_3d = all(('3d' in p['method'] for p in self.history[eyeid]))

        if use_3d:
            data = [(pp['theta'], pp['phi']) for pp in self.history[eyeid]]
        else:
            data = [pp['norm_pos'] for pp in self.history[eyeid]]

        # euklidian distance:
        delta = (data[0][0] - data[1][0])**2, (data[0][1] - data[1][1])**2
        dist = degrees(sqrt(sum(delta)))

        if use_3d and dist > self.dispersion_3d:
            print(use_3d, dist)
        elif not use_3d:
            print(use_3d, dist)

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Saccade Detector'
        self.menu.append(ui.Slider('dispersion_3d', self, label='Dispersion 3D [deg]', min=0.5, max=10.))

    def deinit_ui(self):
        self.remove_menu()
