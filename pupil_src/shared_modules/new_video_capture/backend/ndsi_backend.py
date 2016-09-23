'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import ndsi

from . import Base_Backend
from ..source import NDSI_Source, Fake_Source

import logging, traceback as tb
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Backend(Base_Backend):
    """docstring for NDSI_Backend"""
    def __init__(self, g_pool):
        super(NDSI_Backend, self).__init__(g_pool)
        self.network = ndsi.Network(callbacks=(self.on_event,))
        self.network.start()

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Remote Pupil Mobile sources'))

        def dev_selection_list():
            default = (None, 'Select to activate')
            self.poll_events()
            devices = [
                {
                    'name': str('%s @ %s'%(s['sensor_name'],s['host_name'])),
                    'uid' : s['sensor_uuid']
                }
                for s in self.network.sensors.values()
                if s['sensor_type'] == 'video'
            ]
            dev_pairs = [default] + [(d['uid'], d['name']) for d in devices]
            return zip(*dev_pairs)

        def activate(source_uid):
            if not source_uid:
                return
            settings = {
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'source_id': source_uid,
                'network': self.network
            }
            self.activate_source(NDSI_Source, settings)

        ui_elements.append(ui.Selector(
            'selected_source',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Activate source'
        ))
        self.g_pool.capture_selector_menu.extend(ui_elements)

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def update(self, frame, events):
        self.poll_events()

    def on_event(self, caller, event):
        if event['subject'] == 'detach':
            name = str('%s @ %s'%(event['sensor_name'],event['host_name']))
            self.notify_all({
                'subject': 'capture_backend.source_lost',
                'source_class_name': NDSI_Source.class_name(),
                'source_id': event['sensor_uuid'],
                'name': name
            })
        elif event['subject'] == 'attach':
            name = str('%s @ %s'%(event['sensor_name'],event['host_name']))
            self.notify_all({
                'subject': 'capture_backend.source_found',
                'source_class_name': NDSI_Source.class_name(),
                'source_id': event['sensor_uuid'],
                'name': name
            })

    def source_init_arguments(self):
        """Provides non-serializable init arguments"""
        return {'network': self.network}
