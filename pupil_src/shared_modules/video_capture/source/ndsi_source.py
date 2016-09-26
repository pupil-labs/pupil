'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . import InitialisationError, Base_Source
from ndsi import StreamError

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Source(Base_Source):
    """docstring for NDSI_Source"""
    def __init__(self, g_pool, frame_size, frame_rate, network=None, source_id=None, host_name=None, sensor_name=None,**settings):
        if not network: raise InitialisationError()
        super(NDSI_Source, self).__init__(g_pool)
        self.sensor = None
        try:
            # uuid given
            if source_id:
                self.sensor = network.sensor(source_id, callbacks=(self.on_notification,))
            # host/sensor name combination, prob. from settings
            elif host_name and sensor_name:
                for sensor in network.sensors.values():
                    if (sensor['host_name'] == host_name and
                        sensor['sensor_name'] == sensor_name):
                        self.sensor = network.sensor(sensor['sensor_uuid'], callbacks=(self.on_notification,))
            else:
                for sensor_id in network.sensors:
                    try:
                        self.sensor = network.sensor(sensor_id, callbacks=(self.on_notification,))
                        break
                    except ValueError:
                        continue
        except ValueError:  raise InitialisationError()
        if not self.sensor: raise InitialisationError()

        logger.debug('NDSI Source Sensor: %s'%self.sensor)
        self.control_id_ui_mapping = {}
        self.get_frame_timeout = 1000
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.has_ui = False

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
        if self.has_ui and event['control_id'] not in self.control_id_ui_mapping:
            self.update_control_menu()

    @property
    def frame_size(self):
        return (1280, 720)
    @frame_size.setter
    def frame_size(self,new_size):
        # Subclasses need to call this:
        self.g_pool.on_frame_size_change(new_size)
        # eye.py sets a custom `on_frame_size_change` callback
        # which recalculates the size of the ROI. If this does not
        # happen, the eye process will crash.
    def set_frame_size(self,new_size):
        self.frame_size = new_size

    @property
    def frame_rate(self):
        return 30
    @frame_rate.setter
    def frame_rate(self,new_rate):
        pass

    @property
    def jpeg_support(self):
        raise NotImplementedError()

    @property
    def settings(self):
        settings = super(NDSI_Source, self).settings
        settings['name'] = self.name
        settings['sensor_name'] = self.sensor.name
        settings['host_name'] = self.sensor.host_name
        settings['frame_rate'] = self.frame_rate
        settings['frame_size'] = self.frame_size
        return settings

    @settings.setter
    def settings(self,settings):
        self.frame_size = settings['frame_size']
        self.frame_rate = settings['frame_rate']

    def init_gui(self):
        self.has_ui = True
        self.update_control_menu()

    def update_control_menu(self):
        from pyglui import ui
        del self.g_pool.capture_source_menu.elements[:]
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
                self.g_pool.capture_source_menu.append(ctrl_ui)
        self.g_pool.capture_source_menu.append(ui.Button("Reset to default values",self.sensor.reset_all_control_values))

    def cleanup(self):
        self.sensor = None

    @staticmethod
    def error_class():
        return StreamError