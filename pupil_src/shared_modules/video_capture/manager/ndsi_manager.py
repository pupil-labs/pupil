'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import ndsi

from . import Base_Manager
from ..source import NDSI_Source, Fake_Source

import logging, traceback as tb
logger = logging.getLogger(__name__)

class NDSI_Manager(Base_Manager):
    """Enumerates and activates Pupil Mobile video sources

    Attributes:
        network (ndsi.Network): NDSI Network backend
        selected_host (unicode): Selected host uuid
    """

    gui_name = 'Pupil Mobile'

    def __init__(self, g_pool):
        super(NDSI_Manager, self).__init__(g_pool)
        self.network = ndsi.Network(callbacks=(self.on_event,))
        self.network.start()
        self.selected_host = None

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Remote Pupil Mobile sources'))

        def host_selection_list():
            self.poll_events()
            devices = {
                s['host_uuid']: str(s['host_name']) # removes duplicates
                for s in self.network.sensors.values()
                if s['sensor_type'] == 'video'
            }
            devices = [pair for pair in devices.iteritems()] # create tuples
            # split tuples into 2 lists
            return zip(*(devices or [(None, 'No hosts found')]))

        #def host_selection_getter():

        def view_host(host_uuid):
            if self.selected_host != host_uuid:
                self.selected_host = host_uuid
                self.re_build_ndsi_menu()

        ui_elements.append(ui.Selector(
            'selected_host',self,
            selection_getter=host_selection_list,
            # getter=host_selection_getter,
            setter=view_host,
            label='Remote host'
        ))

        self.g_pool.capture_selector_menu.extend(ui_elements)
        if not self.selected_host: return
        ui_elements = []

        host_menu = ui.Growing_Menu('Remote Host Information')
        ui_elements.append(host_menu)

        def source_selection_list():
            default = (None, 'Select to activate')
            self.poll_events()
            sources = [default] + [
                (s['sensor_uuid'], str(s['sensor_name']))
                for s in self.network.sensors.values()
                if (s['sensor_type'] == 'video' and
                    s['host_uuid'] == self.selected_host)
            ]
            return zip(*sources)

        def activate(source_uid):
            if not source_uid:
                return
            settings = {
                'source_class_name': NDSI_Source.class_name(),
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'source_id': source_uid
            }
            self.activate_source(NDSI_Source, settings)


        host_menu.append(ui.Selector(
            'selected_source',
            selection_getter=source_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Activate source'
        ))

        self.g_pool.capture_selector_menu.extend(ui_elements)

    def re_build_ndsi_menu(self):
        self.deinit_gui()
        self.init_gui()

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def update(self, frame, events):
        self.poll_events()

    def on_event(self, caller, event):
        if event['subject'] == 'detach':
            name = str('%s @ %s'%(event['sensor_name'],event['host_name']))
            self.notify_all({
                'subject': 'capture_manager.source_lost',
                'source_class_name': NDSI_Source.class_name(),
                'source_id': event['sensor_uuid'],
                'name': name
            })
            sensors = self.network.sensors
            if self.selected_host == event['host_uuid']:
                if sensors:
                    any_key = sensors.keys()[0]
                    self.selected_host = sensors[any_key]['host_uuid']
                else:
                    self.selected_host = None
                self.re_build_ndsi_menu()

        elif event['subject'] == 'attach':
            name = str('%s @ %s'%(event['sensor_name'],event['host_name']))
            self.notify_all({
                'subject': 'capture_manager.source_found',
                'source_class_name': NDSI_Source.class_name(),
                'source_id': event['sensor_uuid'],
                'name': name
            })
            if not self.selected_host:
                self.selected_host = event['host_uuid']
                self.re_build_ndsi_menu()

    def source_init_arguments(self):
        """Provides non-serializable init arguments"""
        return {'network': self.network}
