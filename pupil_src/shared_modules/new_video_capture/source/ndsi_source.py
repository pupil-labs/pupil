'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . import Fake_Source

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Source(Fake_Source):
    """docstring for NDSI_Source"""
    def __init__(self, g_pool, network, source_id):
        super(NDSI_Source, self).__init__(g_pool)
        self.sensor = network.sensor(source_id, callbacks=(self.on_notification,))
        logger.debug('NDSI Source Sensor: %s'%self.sensor)

    @property
    def name(self):
        return '%s @ %s'%(self.sensor.name, self.sensor.host_name)

    def poll_events(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def get_frame(self):
        self.poll_events()
        return super(NDSI_Source, self).get_frame()

    def on_notification(self, sensor, event):
        logger.debug('%s: %s'%(sensor,event))