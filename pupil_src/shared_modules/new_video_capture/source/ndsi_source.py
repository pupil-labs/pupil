'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from base_source import Base_Source
from ndsi import StreamError

from pyglui import ui

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Source(Base_Source):
    """docstring for NDSI_Source"""
    def __init__(self, g_pool, network, source_id):
        super(NDSI_Source, self).__init__(g_pool)
        self.sensor = network.sensor(source_id, callbacks=(self.on_notification,))
        logger.debug('NDSI Source Sensor: %s'%self.sensor)
        self.control_menu = None
        self.control_id_ui_mapping = {}
        self.get_frame_timeout = 1000

    @property
    def name(self):
        return '%s @ %s'%(self.sensor.name, self.sensor.host_name)

    def poll_notifications(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def get_frame(self):
        self.poll_notifications()
        return self.sensor.get_newest_data_frame(timeout=self.get_frame_timeout)

    def get_frame_robust(self):
        '''Mirrors uvc.Capture.get_frame_robust()'''
        attempts = 3
        for a in range(attempts):
            try:
                frame = self.get_frame()
            except Exception as e:
                if a:
                    logger.info('Could not get Frame: "%s". Attempt:%s/%s '%(e.message,a+1,attempts))
                else:
                    logger.debug('Could not get Frame of first try: "%s". Attempt:%s/%s '%(e.message,a+1,attempts))
            else:
                return frame
        raise StreamError("Could not grab frame after 3 attempts. Giving up.")

    def on_notification(self, sensor, event):
        if self.control_menu and event['control_id'] not in self.control_id_ui_mapping:
            logger.debug('! update_control_menu call ! %s > %s'%(sensor,event))
            self.update_control_menu()

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
        self.control_menu = ui.Growing_Menu(label='%s Controls'%str(self.sensor.name))
        self.update_control_menu()
        self.parent_menu.append(self.control_menu)

    def update_control_menu(self):
        del self.control_menu.elements[:]
        self.control_id_ui_mapping = {}

        # closure factory
        def make_value_change_fn(ctrl_id):
            def initiate_value_change(val):
                logger.debug('%s: %s >> %s'%(self.sensor, ctrl_id, val))
                self.sensor.set_control_value(ctrl_id, val)
            return initiate_value_change

        for ctrl_id, ctrl_dict in self.sensor.controls.iteritems():
            dtype = ctrl_dict['dtype']
            ctrl_ui = None
            if dtype == "string":
                logger.debug('Text input for %s named "%s"'%(dtype,ctrl_dict['caption']))
                ctrl_ui = ui.Text_Input(
                    'value',
                    ctrl_dict,
                    label=str(ctrl_dict['caption']),
                    setter=make_value_change_fn(ctrl_id))
            elif dtype == "integer" or dtype == "float":
                convert_fn = int if dtype == "integer" else float
                ctrl_ui = ui.Slider(
                    'value',
                    ctrl_dict,
                    label=str(ctrl_dict['caption']),
                    min =convert_fn(ctrl_dict['min'] or 0),
                    max =convert_fn(ctrl_dict['max'] or 100),
                    step=convert_fn(ctrl_dict['res'] or 0.),
                    setter=make_value_change_fn(ctrl_id))
            elif dtype == "bool":
                ctrl_ui = ui.Switch(
                    'value',
                    ctrl_dict,
                    label=str(ctrl_dict['caption']),
                    on_val=ctrl_dict['max'],
                    off_val=ctrl_dict['min'],
                    setter=make_value_change_fn(ctrl_id))
            elif dtype == "selector":
                desc_list = ctrl_dict['selector']
                labels    = [str(desc['caption']) for desc in desc_list]
                selection = [desc['value']        for desc in desc_list]
                ctrl_ui = ui.Selector(
                    'value',
                    ctrl_dict,
                    label=str(ctrl_dict['caption']),
                    selection=selection,
                    labels=labels,
                    setter=make_value_change_fn(ctrl_id))
            if ctrl_ui:
                self.control_id_ui_mapping[ctrl_id] = ctrl_ui
                self.control_menu.append(ctrl_ui)
        self.control_menu.append(ui.Button("Reset to default values",self.sensor.reset_all_control_values))

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