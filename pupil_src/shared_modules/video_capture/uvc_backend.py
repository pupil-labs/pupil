'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import time
import logging
import uvc
from version_utils import VersionFormat
from .base_backend import InitialisationError, Base_Source, Base_Manager

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(uvc.__version__) >= VersionFormat('0.11')

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UVC_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes uvc.Capture:
    """
    def __init__(self, g_pool, frame_size, frame_rate, name=None, preferred_names=(), uid=None, uvc_controls={}):
        super().__init__(g_pool)
        self.uvc_capture = None
        self._restart_in = 3
        assert name or preferred_names or uid

        self.devices = uvc.Device_List()

        devices_by_name = {dev['name']: dev for dev in self.devices}

        # if uid is supplied we init with that
        if uid:
            try:
                self.uvc_capture = uvc.Capture(uid)
            except uvc.OpenError:
                logger.warning("No avalilable camera found that matched {}".format(preferred_names))
            except uvc.InitError:
                logger.error("Camera failed to initialize.")
            except uvc.DeviceNotFoundError:
                logger.warning("No camera found that matched {}".format(preferred_names))

        # otherwise we use name or preffered_names
        else:
            if name:
                preferred_names = (name,)
            else:
                pass
            assert preferred_names

            # try to init by name
            for name in preferred_names:
                for d_name in devices_by_name.keys():
                    if name in d_name:
                        uid_for_name = devices_by_name[d_name]['uid']
                        try:
                            self.uvc_capture = uvc.Capture(uid_for_name)
                        except uvc.OpenError:
                            logger.info("{} matches {} but is already in use or blocked.".format(uid_for_name, name))
                        except uvc.InitError:
                            logger.error("Camera failed to initialize.")
                        else:
                            break

        # check if we were sucessfull
        if not self.uvc_capture:
            logger.error("Init failed. Capture is started in ghost mode. No images will be supplied.")
            self.name_backup = preferred_names
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate

        else:
            self.configure_capture(frame_size, frame_rate, uvc_controls)
            self.name_backup = (self.name,)
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate

    def configure_capture(self, frame_size, frame_rate, uvc_controls):
        # Set camera defaults. Override with previous settings afterwards
        if 'C930e' in self.uvc_capture.name:
                logger.debug('Timestamp offset for c930 applied: -0.1sec')
                self.ts_offset = -0.1
        else:
            self.ts_offset = 0.0

        # UVC setting quirks:
        controls_dict = dict([(c.display_name, c) for c in self.uvc_capture.controls])

        self.frame_size = frame_size
        self.frame_rate = frame_rate
        for c in self.uvc_capture.controls:
            try:
                c.value = uvc_controls[c.display_name]
            except KeyError:
                logger.debug('No UVC setting "{}" found from settings.'.format(c.display_name))

        try:
            controls_dict['Auto Focus'].value = 0
        except KeyError:
            pass

        if ("Pupil Cam1" in self.uvc_capture.name or
            "USB2.0 Camera" in self.uvc_capture.name):

            if ("ID0" in self.uvc_capture.name or "ID1" in self.uvc_capture.name):

                self.uvc_capture.bandwidth_factor = 1.3

                try: controls_dict['Auto Exposure Priority'].value = 0
                except KeyError: pass

                try: controls_dict['Auto Exposure Mode'].value = 1
                except KeyError: pass

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

    def _re_init_capture(self, uid):
        current_size = self.uvc_capture.frame_size
        current_fps = self.uvc_capture.frame_rate
        current_uvc_controls = self._get_uvc_controls()
        self.deinit_gui()
        self.uvc_capture.close()
        self.uvc_capture = uvc.Capture(uid)
        self.configure_capture(current_size, current_fps, current_uvc_controls)
        self.init_gui()

    def _init_capture(self, uid):
        self.deinit_gui()
        self.uvc_capture = uvc.Capture(uid)
        self.configure_capture(self.frame_size_backup, self.frame_rate_backup, self._get_uvc_controls())
        self.init_gui()

    def _re_init_capture_by_names(self, names):
        # burn-in test specific. Do not change text!
        self.devices.update()
        for d in self.devices:
            for name in names:
                if d['name'] == name:
                    logger.info("Found device. {}.".format(name))
                    if self.uvc_capture:
                        self._re_init_capture(d['uid'])
                    else:
                        self._init_capture(d['uid'])
                    return
        raise InitialisationError('Could not find Camera {} during re initilization.'.format(names))

    def _restart_logic(self):
        if self._restart_in <= 0:
            if self.uvc_capture:
                logger.warning("Capture failed to provide frames. Attempting to reinit.")
                self.name_backup = (self.uvc_capture.name,)
                self.uvc_capture = None
            try:
                self._re_init_capture_by_names(self.name_backup)
            except (InitialisationError, uvc.InitError):
                time.sleep(0.02)
                self.deinit_gui()
                self.init_gui()
            self._restart_in = int(5/0.02)
        else:
            self._restart_in -= 1

    def recent_events(self, events):
        try:
            frame = self.uvc_capture.get_frame(0.05)
            frame.timestamp = self.g_pool.get_timestamp()+self.ts_offset
        except uvc.StreamError:
            self._recent_frame = None
            self._restart_logic()
        except (AttributeError, uvc.InitError):
            self._recent_frame = None
            time.sleep(0.02)
            self._restart_logic()
        else:
            self._recent_frame = frame
            events['frame'] = frame
            self._restart_in = 3

    def _get_uvc_controls(self):
        d = {}
        if self.uvc_capture:
            for c in self.uvc_capture.controls:
                d[c.display_name] = c.value
        return d

    def get_init_dict(self):
        d = super().get_init_dict()
        d['frame_size'] = self.frame_size
        d['frame_rate'] = self.frame_rate
        if self.uvc_capture:
            d['name'] = self.name
            d['uvc_controls'] = self._get_uvc_controls()
        else:
            d['preferred_names'] = self.name_backup
        return d

    @property
    def name(self):
        if self.uvc_capture:
            return self.uvc_capture.name
        else:
            return "Ghost capture"

    @property
    def frame_size(self):
        if self.uvc_capture:
            return self.uvc_capture.frame_size
        else:
            return self.frame_size_backup

    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size
        sizes = [abs(r[0]-new_size[0]) for r in self.uvc_capture.frame_sizes]
        best_size_idx = sizes.index(min(sizes))
        size = self.uvc_capture.frame_sizes[best_size_idx]
        if tuple(size) != tuple(new_size):
            logger.warning("%s resolution capture mode not available. Selected {}.".format(new_size, size))
        self.uvc_capture.frame_size = size
        self.frame_size_backup = size

    @property
    def frame_rate(self):
        if self.uvc_capture:
            return self.uvc_capture.frame_rate
        else:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        # closest match for rate
        rates = [abs(r-new_rate) for r in self.uvc_capture.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.uvc_capture.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning("{}fps capture mode not available at ({}) on '{}'. Selected {}fps. ".format(
                new_rate, self.uvc_capture.frame_size, self.uvc_capture.name, rate))
        self.uvc_capture.frame_rate = rate
        self.frame_rate_backup = rate

    @property
    def jpeg_support(self):
        return True

    @property
    def online(self):
        return bool(self.uvc_capture)

    def init_gui(self):
        from pyglui import ui
        ui_elements = []

        # lets define some  helper functions:
        def gui_load_defaults():
            for c in self.uvc_capture.controls:
                try:
                    c.value = c.def_val
                except:
                    pass

        def gui_update_from_device():
            for c in self.uvc_capture.controls:
                c.refresh()

        def set_frame_size(new_size):
            self.frame_size = new_size

        if self.uvc_capture is None:
            ui_elements.append(ui.Info_Text('Capture initialization failed.'))
            self.g_pool.capture_source_menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text('{} Controls'.format(self.name)))
        sensor_control = ui.Growing_Menu(label='Sensor Settings')
        sensor_control.append(ui.Info_Text("Do not change these during calibration or recording!"))
        sensor_control.collapsed = False
        image_processing = ui.Growing_Menu(label='Image Post Processing')
        image_processing.collapsed = True

        sensor_control.append(ui.Selector(
            'frame_size', self,
            setter=set_frame_size,
            selection=self.uvc_capture.frame_sizes,
            label='Resolution'
        ))

        def frame_rate_getter():
            return (self.uvc_capture.frame_rates, [str(fr) for fr in self.uvc_capture.frame_rates])
        sensor_control.append(ui.Selector('frame_rate', self, selection_getter=frame_rate_getter, label='Frame rate'))

        for control in self.uvc_capture.controls:
            c = None
            ctl_name = control.display_name

            # now we add controls
            if control.d_type == bool:
                c = ui.Switch('value', control, label=ctl_name, on_val=control.max_val, off_val=control.min_val)
            elif control.d_type == int:
                c = ui.Slider('value', control, label=ctl_name, min=control.min_val, max=control.max_val, step=control.step)
            elif type(control.d_type) == dict:
                selection = [value for name, value in control.d_type.items()]
                labels = [name for name, value in control.d_type.items()]
                c = ui.Selector('value', control, label=ctl_name, selection=selection, labels=labels)
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

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
        if self.uvc_capture:
            self.uvc_capture.close()
            self.uvc_capture = None
        super().cleanup()


class UVC_Manager(Base_Manager):
    """Manages local USB sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """
    gui_name = 'Local USB'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.devices = uvc.Device_List()

    def get_init_dict(self):
        return {}

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
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'uid': source_uid
            }
            if self.g_pool.process == 'world':
                self.notify_all({'subject': 'start_plugin', "name": "UVC_Source", 'args': settings})
            else:
                self.notify_all({'subject': 'start_eye_capture', 'target': self.g_pool.process, "name": "UVC_Source", 'args': settings})

        ui_elements.append(ui.Selector(
            'selected_source',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Activate source'
        ))
        self.g_pool.capture_selector_menu.extend(ui_elements)

    def cleanup(self):
        self.deinit_gui()
        self.devices.cleanup()
        self.devices = None

    def recent_events(self, events):
        pass
