'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import ndsi

from base_backend import Base_Backend
from ..source import NDSI_Source

import logging, traceback as tb
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Backend(Base_Backend):
    """docstring for NDSI_Backend"""
    def __init__(self, g_pool, settings):
        super(NDSI_Backend, self).__init__(g_pool, settings, should_load_settings=True)
        self.network = ndsi.Network(callbacks=(self.on_event,))
        self.network.start()

    @staticmethod
    def stream_error_class():
        return ndsi.StreamError

    @staticmethod
    def source_type():
        return 'Pupil Mobile'

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def get_frame(self):
        self.poll_events()
        return self.active_source.get_frame_robust()

    def on_event(self, caller, event):
        if (event['subject'] == 'detach' and
            isinstance(self.active_source, NDSI_Source) and
            self.active_source.sensor.uuid == event['sensor_uuid']):
            self.set_active_source_with_id(None, self.active_source.settings)

    def list_sources(self):
        self.poll_events()
        return [
            {
                'name': str('%s @ %s'%(s['sensor_name'],s['host_name'])),
                'uid' : s['sensor_uuid']
            }
            for s in self.network.sensors.values()
            if s['sensor_type'] == 'video'
        ]

    def set_active_source_with_name(self, name, settings=None):
        if self.active_source:
            settings = settings or self.active_source.settings
        succesfull = super(NDSI_Backend,self).set_active_source_with_name(name,settings)
        if not succesfull:
            for dev in self.list_sources():
                if dev['name'] == name:
                    return self.set_active_source_with_id(dev['uid'], settings=settings)
            return self.set_active_source_with_id(None)
        return succesfull

    def set_active_source_with_id(self, source_id, settings=None):
        succesfull = super(NDSI_Backend,self).set_active_source_with_id(source_id, settings)
        if not succesfull:
            if self.active_source:
                settings = settings or self.active_source.settings
                self.active_source.close()

            try:
                self.active_source = NDSI_Source(self.g_pool, self.network, source_id)
                self.active_source_id = source_id
                if settings:
                    self.active_source.settings = settings
                self.active_source.init_gui(self.menu)
            except Exception as e:
                tb.print_exc()
                logger.error('Initializing Pupil Mobile source failed because of: %s'%str(e))
                return False

