'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import logging
import time

import pyrealsense as pyrs
from version_utils import VersionFormat
from .base_backend import InitialisationError, Base_Source, Base_Manager

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(pyrs.__version__) >= VersionFormat('2.1')

# logging
logging.getLogger('pyrealsense').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ColorFrame(object):
    def __init__(self, device):
        self._rgb = device.color


class Realsense_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes pyrs.Device:
    """
    def __init__(self, g_pool, device_id, frame_size, frame_rate, uvc_controls={}):
        super().__init__(g_pool)
        self.device = None
        self.service = pyrs.Service()
        self._initialize_device(device_id)

    def _initialize_device(self, device_id, settings=None):
        devices = tuple(self.service.get_devices())
        if not devices:
            logger.error("Camera failed to initialize. No cameras connected.")

        if device_id >= len(devices):
            logger.error("Camera with id {} not found. Initializing default camera.".format(device_id))
            device_id = 0

        self.device = self.service.Device(device_id)
        if not settings:
            self.device.apply_ivcam_preset(0)

    def cleanup(self):
        self.service.stop()

    def get_frame(self, frame_cls, timeout, poll_interval=0.01):
        start = time.time()
        while not self.device.poll_for_frame():
            if time.time() - start > timeout:
                raise TimeoutError()
            time.sleep(poll_interval)
        return frame_cls(self.device)

    def recent_events(self, events):
        try:
            frame = self.get_frame(ColorFrame, 0.05)
            frame.timestamp = self.g_pool.get_timestamp()+self.ts_offset
        except TimeoutError:
            self._recent_frame = None
            # react to timeout
        except pyrs.RealsenseError as err:
            self._recent_frame = None
            # act according to err.function
            # self._restart_logic()
        else:
            self._recent_frame = frame
            events['frame'] = frame
            self._restart_in = 3


class Realsense_Manager(Base_Manager):
    """Manages Intel RealSense 3D sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """
    gui_name = 'RealSense 3D'

    def get_init_dict(self):
        return {}

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Intel RealSense 3D sources'))

        def pair(d):
            fmt = '- ' if d['is_streaming'] else ''
            fmt += d['name']
            return d['id'], fmt

        def dev_selection_list():
            default = (None, 'Select to activate')
            try:
                with pyrs.Service() as service:
                    dev_pairs = [default] + [pair(d) for d in service.get_devices()]
            except pyrs.RealsenseError:
                dev_pairs = [default]

            return zip(*dev_pairs)

        def activate(source_uid):
            if not source_uid:
                return

            with pyrs.Service as service:
                if not service.is_device_streaming(source_uid):
                    logger.error("The selected camera is already in use or blocked.")
                    return
            settings = {
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'device_id': source_uid
            }
            if self.g_pool.process == 'world':
                self.notify_all({'subject': 'start_plugin',
                                 'name': 'Realsense_Source',
                                 'args': settings})
            else:
                self.notify_all({'subject': 'start_eye_capture',
                                 'target': self.g_pool.process,
                                 'name': 'Realsense_Source',
                                 'args': settings})

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
