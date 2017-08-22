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
from plugin import Plugin

import logging
logger = logging.getLogger(__name__)


class Calibration_Plugin(Plugin):
    '''base class for all calibration routines'''
    uniqueness = 'by_base_class'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.active_calibration_plugin = self
        self.pupil_confidence_threshold = 0.6
        self.active = False
        self.mode = 'calibration'

    @property
    def mode_pretty(self):
        return self.mode.replace('_', ' ').title()

    def on_notify(self, notification):
        '''Handles calibration notifications

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``: Stops the calibration procedure

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped
            ``calibration.failed``: Calibration failed
            ``calibration.successful``: Calibration succeeded

        Args:
            notification (dictionary): Notification dictionary
        '''
        if notification['subject'].endswith('.should_start'):
            if self.active:
                logger.warning('{} already running.'.format(self.mode_pretty))
            else:
                if notification['subject'].startswith('calibration'):
                    self.mode = 'calibration'
                elif notification['subject'].startswith('accuracy_test'):
                    self.mode = 'accuracy_test'
                else:
                    return
                self.start()
        elif notification['subject'].endswith('should_stop'):
            if not (notification['subject'].startswith('calibration') or
                    notification['subject'].startswith('accuracy_test')):
                return
            if self.active:
                self.stop()
            else:
                logger.warning('{} already stopped.'.format(self.mode_pretty))

    def init_gui(self):
        self.button = None
        self.calib_button = ui.Thumb('active', self, label='C', setter=self.toggle_calibration, hotkey='c')
        self.test_button = ui.Thumb('active', self, label='T', setter=self.toggle_accuracy_test, hotkey='t')

        on_color = (.3, .2, 1., .9)
        self.calib_button.on_color[:] = on_color
        self.test_button.on_color[:] = on_color

        self.g_pool.quickbar.insert(0, self.calib_button)
        self.g_pool.quickbar.insert(1, self.test_button)

    def deinit_gui(self):
        self.g_pool.quickbar.remove(self.calib_button)
        self.g_pool.quickbar.remove(self.test_button)
        self.calib_button = None
        self.test_button = None

    def toggle_calibration(self, _=None):
        if self.active:
            self.notify_all({'subject': 'calibration.should_stop'})
        else:
            self.notify_all({'subject': 'calibration.should_start'})

    def toggle_accuracy_test(self, _=None):
        if self.active:
            self.notify_all({'subject': 'accuracy_test.should_stop'})
        else:
            self.notify_all({'subject': 'accuracy_test.should_start'})

    def start(self):
        if self.mode == 'calibration':
            self.button = self.calib_button
            self.g_pool.quickbar.remove(self.test_button)
        elif self.mode == 'accuracy_test':
            self.button = self.test_button
            self.g_pool.quickbar.remove(self.calib_button)
        self.notify_all({'subject': 'calibration.started'})

    def stop(self):
        self.button = None  # reset buttons
        if self.calib_button not in self.g_pool.quickbar:
            self.g_pool.quickbar.insert(0, self.calib_button)
        if self.test_button not in self.g_pool.quickbar:
            self.g_pool.quickbar.insert(1, self.test_button)

        self.notify_all({'subject': '{}.stopped'.format(self.mode)})

    def finish_accuracy_test(self, pupil_list, ref_list):
        ts = self.g_pool.get_timestamp()
        self.notify_all({'subject': 'start_plugin', 'name': 'Accuracy_Visualizer'})
        self.notify_all({'subject': 'accuracy_test.data', 'timestamp': ts,
                         'pupil_list': pupil_list, 'ref_list': ref_list, 'record': True})
