'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from .base_backend import InitialisationError, StreamError, Base_Source, Base_Manager
from .fake_backend import Fake_Source

import ndsi
assert ndsi.NDS_PROTOCOL_VERSION >= '0.2.11'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NDSI_Source(Base_Source):
    """Pupil Mobile video source

    Attributes:
        get_frame_timeout (float): Maximal waiting time for next frame
        sensor (ndsi.Sensor): NDSI sensor backend
    """
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
        if not self.sensor.supports_data_subscription:
            self.cleanup()
            raise InitialisationError('Source does not support data subscription.')

        logger.debug('NDSI Source Sensor: %s'%self.sensor)
        self.control_id_ui_mapping = {}
        self.get_frame_timeout = 2000
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.has_ui = False

    @property
    def name(self):
        return '%s @ %s'%(self.sensor.name, self.sensor.host_name)

    def poll_notifications(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def _get_frame(self):
        self.poll_notifications()
        return self.sensor.get_newest_data_frame(timeout=self.get_frame_timeout)

    def get_frame(self):
        '''Mirrors uvc.Capture.get_frame_robust()'''
        attempts = 3
        for a in range(attempts):
            try:
                frame = self._get_frame()
            except Exception as e:
                # check if streaming is enabled
                if 'streaming' in self.sensor.controls:
                    pub_ctrl = self.sensor.controls['streaming']
                    if not pub_ctrl.get('value', False):
                        self.sensor.set_control_value('streaming', True)
                if a:
                    logger.info('Could not get Frame: "%s". Attempt:%s/%s '%(e.message,a+1,attempts))
                else:
                    logger.debug('Could not get Frame of first try: "%s". Attempt:%s/%s '%(e.message,a+1,attempts))
            else:
                return frame
        raise StreamError("Could not grab frame after 3 attempts. Giving up.")

    def on_notification(self, sensor, event):
        if event['subject'] == 'error':
            logger.warning('Error: %s'%event['error_str'])
        elif self.has_ui and event['control_id'] not in self.control_id_ui_mapping:
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
        return True

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

        for ctrl_id, ctrl_dict in self.sensor.controls.items():
            try:
                dtype = ctrl_dict['dtype']
                ctrl_ui = None
                if dtype == "string":
                    ctrl_ui = ui.Text_Input(
                        'value',
                        ctrl_dict,
                        label=unicode(ctrl_dict['caption']),
                        setter=make_value_change_fn(ctrl_id))
                elif dtype == "integer" or dtype == "float":
                    convert_fn = int if dtype == "integer" else float
                    ctrl_ui = ui.Slider(
                        'value',
                        ctrl_dict,
                        label=unicode(ctrl_dict['caption']),
                        min =convert_fn(ctrl_dict.get('min', 0)),
                        max =convert_fn(ctrl_dict.get('max', 100)),
                        step=convert_fn(ctrl_dict.get('res', 0.)),
                        setter=make_value_change_fn(ctrl_id))
                elif dtype == "bool":
                    ctrl_ui = ui.Switch(
                        'value',
                        ctrl_dict,
                        label=unicode(ctrl_dict['caption']),
                        on_val=ctrl_dict.get('max',True),
                        off_val=ctrl_dict.get('min',False),
                        setter=make_value_change_fn(ctrl_id))
                elif dtype == "selector":
                    desc_list = ctrl_dict['selector']
                    labels    = [unicode(desc['caption']) for desc in desc_list]
                    selection = [desc['value']        for desc in desc_list]
                    ctrl_ui = ui.Selector(
                        'value',
                        ctrl_dict,
                        label=unicode(ctrl_dict['caption']),
                        selection=selection,
                        labels=labels,
                        setter=make_value_change_fn(ctrl_id))
                if ctrl_ui:
                    self.control_id_ui_mapping[ctrl_id] = ctrl_ui
                    self.g_pool.capture_source_menu.append(ctrl_ui)
            except:
                logger.error('Exception for control:\n%s'%ctrl_dict)
                import traceback as tb
                tb.print_exc()
        self.g_pool.capture_source_menu.append(ui.Button("Reset to default values",self.sensor.reset_all_control_values))

    def cleanup(self):
        self.sensor.unlink()
        self.sensor = None

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

    def cleanup(self):
        self.deinit_gui()
        self.network.stop()

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Remote Pupil Mobile sources'))

        def host_selection_list():
            devices = {
                s['host_uuid']: s['host_name'] # removes duplicates
                for s in self.network.sensors.values()
                if s['sensor_type'] == 'video'
            }
            devices = [pair for pair in devices.items()] # create tuples
            # split tuples into 2 lists
            return zip(*(devices or [(None, 'No hosts found')]))

        def view_host(host_uuid):
            if self.selected_host != host_uuid:
                self.selected_host = host_uuid
                self.re_build_ndsi_menu()

        host_sel, host_sel_labels = host_selection_list()
        ui_elements.append(ui.Selector(
            'selected_host',self,
            selection=host_sel,
            labels=host_sel_labels,
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
            #self.poll_events()
            sources = [default] + [
                (s['sensor_uuid'], s['sensor_name'])
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
            self.activate_source(settings)

        src_sel, src_sel_labels = source_selection_list()
        host_menu.append(ui.Selector(
            'selected_source',
            selection=src_sel,
            labels=src_sel_labels,
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
            logger.debug('detached: %s'%event)
            name = str('%s @ %s'%(event['sensor_name'],event['host_name']))
            self.notify_all({
                'subject': 'capture_manager.source_lost',
                'source_class_name': NDSI_Source.class_name(),
                'source_id': event['sensor_uuid'],
                'name': name
            })
            sensors = [s for s in self.network.sensors.values() if s['sensor_type'] == 'video']
            if self.selected_host == event['host_uuid']:
                if sensors:
                    self.selected_host = sensors[0]['host_uuid']
                else:
                    self.selected_host = None
                self.re_build_ndsi_menu()

        elif (event['subject'] == 'attach' and
            event['sensor_type'] == 'video'):
            logger.debug('attached: %s'%event)
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

    def activate_source(self, settings={}):
        try:
            capture = NDSI_Source(self.g_pool,network=self.network, **settings)
        except InitialisationError as init_error:
            logger.error('NDSI source could not be initialised.')
            if init_error.message: logger.error(init_error.message)
            logger.debug('NDSI source init settings:\n\t%s\n\t%s'%(self.network,settings))
        else:
            self.g_pool.capture.deinit_gui()
            self.g_pool.capture.cleanup()
            self.g_pool.capture = None
            self.g_pool.capture = capture
            self.g_pool.capture.init_gui()

    def recover(self):
        if self.g_pool.capture.class_name() == Fake_Source.class_name():
            preferred = self.g_pool.capture.preferred_source
            if preferred['source_class_name'] == NDSI_Source.class_name():
                self.activate_source(preferred)

    def on_notify(self,n):
        """Provides UI for the capture selection

        Reacts to notification:
            ``capture_manager.source_found``: Check if recovery is possible

        Emmits notifications:
            ``capture_manager.source_found``
            ``capture_manager.source_lost``
        """
        if (n['subject'].startswith('capture_manager.source_found')):
            self.recover()
