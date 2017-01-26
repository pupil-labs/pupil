'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from time import strftime, localtime
from pyglui import ui
from plugin import Plugin

import logging
logger = logging.getLogger(__name__)


class Remote_Recorder(Plugin):

    order = .3
    uniqueness = 'by_class'

    def __init__(self, g_pool, session_name='Unnamed session'):
        super().__init__(g_pool)
        self.session_name = session_name
        self.running = False
        self.menu_toggle = None
        self.quickbar_toggle = None
        self.menu = None

    def toggle_recording(self, *args, **kwargs):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if not self.running:
            del self.menu[-1]
            self.menu_toggle = ui.Button('Stop Recording', self.toggle_recording)
            self.menu.append(self.menu_toggle)
            unique_session_name = strftime("%Y%m%d%H%M%S", localtime())
            self.notify_all({
                'subject': 'remote_recording.should_start',
                'session_name': '{}/{}'.format(self.session_name, unique_session_name)})
            self.running = True

    def stop(self):
        if self.running:
            del self.menu[-1]
            self.menu_toggle = ui.Button('Start Recording', self.toggle_recording)
            self.menu.append(self.menu_toggle)
            self.notify_all({'subject': 'remote_recording.should_stop'})
            self.running = False

    def on_notify(self, notification):
        subject = notification['subject']
        if subject == 'ndsi.host_recording.stopped' and self.running:
            source = notification['source']
            logger.warning('Recording on {} was stopped remotely. Stopping whole recording.'.format(source))
            self.stop()

    def init_gui(self):
        self.menu = ui.Growing_Menu('Remote Recorder')
        self.g_pool.sidebar.append(self.menu)
        self.menu.append(ui.Button('Close', self.close))
        self.menu.append(ui.Info_Text('Starts a recording session on each connected Pupil Mobile source.'))
        self.menu.append(ui.Text_Input('session_name', self))
        self.menu_toggle = ui.Button('Start Recording', self.toggle_recording)
        # ↴: Unicode: U+21B4, UTF-8: E2 86 B4
        self.quickbar_toggle = ui.Thumb('running', self, setter=self.toggle_recording,
                                        label=chr(0xf03d), label_font='fontawesome',
                                        label_offset_size=-30, hotkey='e')
        self.quickbar_toggle.on_color[:] = (1, .0, .0, .8)

        self.menu.append(self.menu_toggle)
        self.g_pool.quickbar.append(self.quickbar_toggle)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.g_pool.quickbar.remove(self.quickbar_toggle)
            self.menu = None
            self.menu_toggle = None
            self.quickbar_toggle = None

    def close(self):
        self.alive = False

    def cleanup(self):
        self.stop()
        self.deinit_gui()

    def get_init_dict(self):
        return {'session_name': self.session_name}
