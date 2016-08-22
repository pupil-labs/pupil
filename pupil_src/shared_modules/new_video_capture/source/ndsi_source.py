'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . import Fake_Source

from pyglui import ui

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Source(Fake_Source):
    """docstring for NDSI_Source"""
    def __init__(self, g_pool, network, source_id):
        super(NDSI_Source, self).__init__(g_pool)
        self.sensor = network.sensor(source_id, callbacks=(self.on_notification,))
        logger.debug('NDSI Source Sensor: %s'%self.sensor)
        self.control_menu = None

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
        if self.control_menu: self.update_control_menu()

    def set_frame_size(self,new_size):
        self.frame_size = new_size

    @property
    def settings(self):
        settings = {}
        settings['name'] = self.sensor.name
        settings['frame_rate'] = self.frame_rate
        settings['frame_size'] = self.frame_size
        return settings

    @settings.setter
    def settings(self,settings):
        self.frame_size = settings['frame_size']
        self.frame_rate = settings['frame_rate']

    def init_gui(self, parent_menu):
        self.parent_menu = parent_menu
        #lets define some  helper functions:
        def gui_load_defaults():
            pass

        self.control_menu = ui.Growing_Menu(label='%s Controls'%str(self.sensor.name))
        self.update_control_menu()
        self.parent_menu.append(self.control_menu)

    def update_control_menu(self):
        del self.control_menu.elements[:]
        def print_new_value(val):
            logger.debug('New value: %s'%val)
        for ctrl_id, ctrl_dict in self.sensor.controls.iteritems():
            logger.debug('Creating UI for %s'%ctrl_dict)
            self.control_menu.append(ui.Text_Input(
                'value',
                ctrl_dict,
                label=str(ctrl_dict['caption']),
                setter=print_new_value
                ))


    def deinit_gui(self):
        if self.control_menu:
            del self.control_menu.elements[:]
            self.parent_menu.remove(self.control_menu)
            self.control_menu = None
            self.parent_menu = None

    def close(self):
        self.deinit_gui()
        self.sensor = None

    def __del__(self):
        self.close()