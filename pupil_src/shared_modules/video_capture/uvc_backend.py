'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . base_backend import InitialisationError, StreamError, Base_Source, Base_Manager
from . fake_backend import Fake_Source

import uvc, time
#check versions for our own depedencies as they are fast-changing
assert uvc.__version__ >= '0.8'
from ctypes import c_double

#logging
import logging
logger = logging.getLogger(__name__)

class AvalilabeDeviceNotFoundError(InitialisationError):
    pass


class UVC_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes uvc.Capture:
     - adds UI elements
     - adds timestamping sanitization fns.

    Attributes:
        uvc_capture (uvc.Capture): UVC backend object
    """
    def __init__(self, g_pool, frame_size, frame_rate, name=None, preferred_names=None, uid=None, uvc_controls={}, **settings):
        super(UVC_Source, self).__init__(g_pool)
        self.uvc_capture = None
        self.devices = uvc.Device_List()
        devices_by_name = {dev['name']: dev for dev in self.devices}
        devices_by_uid  = {dev['uid']: dev for dev in self.devices}


        #if uid is supplied we init with that
        if uid:
            try:
                self.uvc_capture = uvc.Capture(uid)
            except uvc.OpenError:
                raise AvalilabeDeviceNotFoundError("No avalilable camera found that matched %s"%preferred_names)
            except uvc.InitError:
                raise InitialisationError("Camera failed to initialize.")
            except uvc.DeviceNotFoundError as e:
                raise AvalilabeDeviceNotFoundError("No camera found that matched %s"%preferred_names)

        #otherwise we use name or preffered_names
        else:
            if name:
                preferred_names = [name]
            else:
                pass

            assert preferred_names

            #try to init by name
            for name in preferred_names:
                if name in devices_by_name:
                    uid_for_name = devices_by_name[name]['uid']
                    try:
                        self.uvc_capture = uvc.Capture(uid_for_name)
                    except uvc.OpenError:
                        logger.info("%s matches %s but is already in use or blocked."%(uid_for_name,name))
                    except uvc.InitError as e:
                        raise InitialisationError("Camera failed to initialize.")
                    else:
                        break
            #check if we where sucessfull
            if not self.uvc_capture:
                raise AvalilabeDeviceNotFoundError("No avalilable camera found that matched %s"%preferred_names)
        self.update_capture(frame_size,frame_rate,uvc_controls)

    def update_capture(self,frame_size,frame_rate,uvc_controls={}):
        # Set camera defaults. Override with previous settings afterwards
        if 'C930e' in self.uvc_capture.name:
                logger.debug('Timestamp offset for c930 applied: -0.1sec')
                self.ts_offset = -0.1
        else:
            self.ts_offset = 0.0

        #UVC setting quirks:
        controls_dict = dict([(c.display_name,c) for c in self.uvc_capture.controls])
        try:
            controls_dict['Auto Focus'].value = 0
        except KeyError:
            pass

        if ("Pupil Cam1" in self.uvc_capture.name or
            "USB2.0 Camera" in self.uvc_capture.name):

            if ("ID0" in self.uvc_capture.name or
                "ID1" in self.uvc_capture.name):

                self.uvc_capture.bandwidth_factor = 1.3

                try: controls_dict['Auto Exposure Priority'].value = 0
                except KeyError: pass

                try: controls_dict['Auto Exposure Mode'].value = 1
                except KeyError as e: pass

                try:controls_dict['Saturation'].value = 0
                except KeyError: pass

                try: controls_dict['Absolute Exposure Time'].value = 63
                except KeyError: pass

                try: controls_dict['Backlight Compensation'].value = 2
                except KeyError: pass

                try: controls_dict['Gamma'].value = 100
                except KeyError: pass

            else:
                self.uvc_capture.bandwidth_factor = 2.0
                try: controls_dict['Auto Exposure Priority'].value = 1
                except KeyError: pass
        else:
            self.uvc_capture.bandwidth_factor = 3.0
            try: controls_dict['Auto Focus'].value = 0
            except KeyError: pass

        self.frame_size = frame_size
        self.frame_rate = frame_rate
        for c in self.uvc_capture.controls:
            try:
                c.value = uvc_controls[c.display_name]
            except KeyError as e:
                logger.debug('No UVC setting "%s" found from settings.'%c.display_name)

    def re_init_capture(self,uid):
        current_size = self.uvc_capture.frame_size
        current_fps = self.uvc_capture.frame_rate
        self.deinit_gui()
        self.uvc_capture.close()
        self.uvc_capture = uvc.Capture(uid)
        self.update_capture(current_size,current_fps)
        self.init_gui()

    def _re_init_capture_by_name(self,name):
        for x in range(4):
            self.devices.update()
            for d in self.devices:
                if d['name'] == name:
                    logger.info("Found device. %s."%name)
                    self.re_init_capture(d['uid'])
                    return
            time.sleep(1.5)
        logger.warning('Could not find Camera %s during re initilization.'%name)
        raise InitialisationError('Could not find Camera %s during re initilization.'%name)

    def get_frame(self):
        try:
            frame = self.uvc_capture.get_frame_robust()
        except uvc.StreamError:
            try:
                self._re_init_capture_by_name(self.uvc_capture.name)
                frame = self.uvc_capture.get_frame_robust()
            except InitialisationError as e:
                raise StreamError(e.message)
            except uvc.InitError as e:
                raise StreamError(str(e))
        except uvc.InitError as e:
            raise StreamError(e.message)
        frame.timestamp = self.g_pool.get_timestamp()+self.ts_offset
        return frame

    @property
    def name(self):
        return self.uvc_capture.name

    @property
    def settings(self):
        settings = super(UVC_Source, self).settings
        settings['frame_rate'] = self.frame_rate
        settings['frame_size'] = self.frame_size
        settings['uvc_controls'] = {}
        for c in self.uvc_capture.controls:
            settings['uvc_controls'][c.display_name] = c.value
        return settings
    @settings.setter
    def settings(self,settings):
        self.frame_size = settings['frame_size']
        self.frame_rate = settings['frame_rate']
        for c in self.uvc_capture.controls:
            try:
                c.value = settings['uvc_controls'][c.display_name]
            except KeyError as e:
                logger.debug('No UVC setting "%s" found from settings.'%c.display_name)
    @property
    def frame_size(self):
        return self.uvc_capture.frame_size
    @frame_size.setter
    def frame_size(self,new_size):
        self.g_pool.on_frame_size_change(new_size)
        #closest match for size
        sizes = [ abs(r[0]-new_size[0]) for r in self.uvc_capture.frame_sizes ]
        best_size_idx = sizes.index(min(sizes))
        size = self.uvc_capture.frame_sizes[best_size_idx]
        if size != new_size:
            logger.warning("%s resolution capture mode not available. Selected %s."%(new_size,size))
        self.uvc_capture.frame_size = size

    def set_frame_size(self,new_size):
        self.frame_size = new_size

    @property
    def frame_rate(self):
        return self.uvc_capture.frame_rate
    @frame_rate.setter
    def frame_rate(self,new_rate):
        #closest match for rate
        rates = [ abs(r-new_rate) for r in self.uvc_capture.frame_rates ]
        best_rate_idx = rates.index(min(rates))
        rate = self.uvc_capture.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning("%sfps capture mode not available at (%s) on '%s'. Selected %sfps. "%(new_rate,self.uvc_capture.frame_size,self.uvc_capture.name,rate))
        self.uvc_capture.frame_rate = rate

    @property
    def jpeg_support(self):
        return True

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        #lets define some  helper functions:
        def gui_load_defaults():
            for c in self.uvc_capture.controls:
                try:
                    c.value = c.def_val
                except:
                    pass
        def gui_update_from_device():
            for c in self.uvc_capture.controls:
                c.refresh()

        ui_elements.append(ui.Info_Text('%s Controls'%self.uvc_capture.name))
        sensor_control = ui.Growing_Menu(label='Sensor Settings')
        sensor_control.append(ui.Info_Text("Do not change these during calibration or recording!"))
        sensor_control.collapsed=False
        image_processing = ui.Growing_Menu(label='Image Post Processing')
        image_processing.collapsed=True

        sensor_control.append(ui.Selector(
            'frame_size',self,
            setter=self.set_frame_size,
            selection=self.uvc_capture.frame_sizes,
            label='Resolution'
        ))
        sensor_control.append(ui.Selector('frame_rate',self, selection=self.uvc_capture.frame_rates,label='Frame rate' ) )


        for control in self.uvc_capture.controls:
            c = None
            ctl_name = control.display_name

            #now we add controls
            if control.d_type == bool :
                c = ui.Switch('value',control,label=ctl_name, on_val=control.max_val, off_val=control.min_val)
            elif control.d_type == int:
                c = ui.Slider('value',control,label=ctl_name,min=control.min_val,max=control.max_val,step=control.step)
            elif type(control.d_type) == dict:
                selection = [value for name,value in control.d_type.items()]
                labels = [name for name,value in control.d_type.items()]
                c = ui.Selector('value',control, label = ctl_name, selection=selection,labels = labels)
            else:
                pass
            # if control['disabled']:
            #     c.read_only = True
            # if ctl_name == 'Exposure, Auto Priority':
            #     # the controll should always be off. we set it to 0 on init (see above)
            #     c.read_only = True

            if c is not None:
                if control.unit == 'processing_unit':
                    image_processing.append(c)
                else:
                    sensor_control.append(c)

        ui_elements.append(sensor_control)
        if image_processing.elements:
            ui_elements.append(image_processing)
        ui_elements.append(ui.Button("refresh",gui_update_from_device))
        ui_elements.append(ui.Button("load defaults",gui_load_defaults))
        self.g_pool.capture_source_menu.extend(ui_elements)

    def stop_stream(self):
        if self.uvc_capture:
            self.uvc_capture.stop_stream()

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
        self.uvc_capture.close()
        self.uvc_capture = None

class UVC_Manager(Base_Manager):
    """Manages local USB sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """
    gui_name = 'Local USB'

    def __init__(self, g_pool, check_devices=True):
        super(UVC_Manager, self).__init__(g_pool)
        # self.last_check_ts = 0.
        # self.check_intervall = 3.5
        # self.check_devices = check_devices
        self.devices = uvc.Device_List()

    def get_init_dict(self):
        return {'check_devices':True}

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Local UVC sources'))

        def dev_selection_list():
            default = (None, 'Select to activate')
            self.devices.update()
            dev_pairs = [default] + [(d['uid'], d['name']) for d in self.devices]
            return zip(*dev_pairs)

        def activate(source_uid):
            if not source_uid:
                return
            if not uvc.is_accessible(source_uid):
                logger.error("The selected camera is already in use or blocked.")
                return
            settings = {
                'source_class_name': UVC_Source.class_name(),
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'uid': source_uid
            }
            self.activate_source(settings)

        ui_elements.append(ui.Selector(
            'selected_source',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Activate source'
        ))
        self.g_pool.capture_selector_menu.extend(ui_elements)

    # def update(self, frame, events):
    #     now = time.time()
    #     if (self.check_devices and
    #         now - self.last_check_ts > self.check_intervall):

    #         self.last_check_ts = now
    #         self.devices.update()
    #         for device in self.devices:
    #             delay = {"Pupil Cam1 ID0":1.5,'Pupil Cam1 ID1':2.5}.get(device['name'],1.)
    #             self.notify_all({
    #                 'subject': 'capture_manager.source_found.%s'%device['name'],
    #                 'source_class_name': UVC_Source.class_name(),
    #                 'name': device['name'],
    #                 'uid': device['uid'],
    #                 'delay':delay
    #             })


    def activate_source(self, settings={}):
        if self.g_pool.capture.class_name() == UVC_Source.class_name():
            self.g_pool.capture.stop_stream()
        try:
            capture = UVC_Source(self.g_pool, **settings)
        except AvalilabeDeviceNotFoundError as e:
            pass
        except InitialisationError as e:
            logger.error(InitialisationError.msg)
            logger.debug('UVC source init settings: %s'%(settings))
        else:
            self.g_pool.capture.deinit_gui()
            self.g_pool.capture.cleanup()
            self.g_pool.capture = None
            self.g_pool.capture = capture
            self.g_pool.capture.init_gui()

    # def recover(self,new_device):
    #     if self.g_pool.capture.class_name() == Fake_Source.class_name():
    #         preferred_settings = self.g_pool.capture.preferred_source
    #         if preferred_settings['source_class_name'] == UVC_Source.class_name():
    #             if preferred_settings['name'] == new_device['name'] or new_device['name'] in preferred_settings.get('preferred_names',[]):
    #                 preferred_settings['uid'] = new_device['uid']
    #                 logger.info("Attempting to auto connect to %s"%new_device['name'])
    #                 self.activate_source(preferred_settings)

    # def on_notify(self,n):
    #     """Provides UI for the capture selection

    #     Reacts to notification:
    #         ``capture_manager.source_found``: Check if recovery is possible

    #     Emmits notifications:
    #         ``capture_manager.source_found``
    #         ``capture_manager.source_lost``
    #     """
    #     if (n['subject'].startswith('capture_manager.source_found')):
    #         self.recover(n)

    def cleanup(self):
        self.deinit_gui()
        self.devices.cleanup()
        self.devices = None