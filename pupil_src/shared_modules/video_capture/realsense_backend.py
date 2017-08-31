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
import cv2
import os

import pyrealsense as pyrs
from pyrealsense.stream import ColorStream, DepthStream, DACStream
from pyrealsense.constants import rs_stream, rs_option

from version_utils import VersionFormat
from .base_backend import Base_Source, Base_Manager
from av_writer import AV_Writer
from camera_models import load_intrinsics

import gl_utils
from pyglui import cygl
import cython_methods
import numpy as np

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(pyrs.__version__) >= VersionFormat('2.2')

# logging
logging.getLogger('pyrealsense').setLevel(logging.ERROR + 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ColorFrame(object):
    def __init__(self, device):
        # we need to keep this since there is no cv2 conversion for our planar format
        self._yuv422 = device.color
        self._shape = self._yuv422.shape[:2]
        self._yuv = np.empty(self._yuv422.size, dtype=np.uint8)
        y_plane = self._yuv422.size//2
        u_plane = y_plane//2
        self._yuv[:y_plane] = self._yuv422[:, :, 0].flatten()
        self._yuv[y_plane:y_plane+u_plane] = self._yuv422[:, ::2, 1].flatten()
        self._yuv[y_plane+u_plane:] = self._yuv422[:, 1::2, 1].flatten()
        self._bgr = None
        self._gray = None

    @property
    def height(self):
        return self._shape[0]

    @property
    def width(self):
        return self._shape[1]

    @property
    def yuv_buffer(self):
        return self._yuv

    @property
    def yuv422(self):
        Y = self._yuv[:self._yuv.size//2]
        U = self._yuv[self._yuv.size//2:3*self._yuv.size//4]
        V = self._yuv[3*self._yuv.size//4:]

        Y.shape = self._shape
        U.shape = self._shape[0], self._shape[1]//2
        V.shape = self._shape[0], self._shape[1]//2

        return Y, U, V

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = cv2.cvtColor(self._yuv422, cv2.COLOR_YUV2BGR_YUVY)
        return self._bgr

    @property
    def img(self):
        return self.bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self._yuv[:self._yuv.size//2]
            self._gray.shape = self._shape
        return self._gray


class DepthFrame(object):
    def __init__(self, device):
        self._bgr = None
        self._gray = None
        self.depth = device.depth
        self.yuv_buffer = None

    @property
    def height(self):
        return self.depth.shape[0]

    @property
    def width(self):
        return self.depth.shape[1]

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = cython_methods.cumhist_color_map16(self.depth)
        return self._bgr

    @property
    def img(self):
        return self.bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.bgr, cv2.cv2.COLOR_BGR2GRAY)
        return self._gray


class Control(object):
    def __init__(self, device, opt_range, value):
        self._dev = device
        self._value = value
        self.range = opt_range
        self.label = rs_option.name_for_value[opt_range.option]
        self.label = self.label.replace('RS_OPTION_', '')
        self.label = self.label.replace('R200_', '')
        self.label = self.label.replace('_', ' ')
        self.label = self.label.title()
        self.description = self._dev.get_device_option_description(opt_range.option)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        try:
            self._dev.set_device_option(self.range.option, val)
        except pyrs.RealsenseError as err:
            logger.error('Setting option "{}" failed'.format(self.label))
            logger.debug('Reason: {}'.format(err))
        else:
            self._value = val

    def refresh(self):
        self._value = self._dev.get_device_option(self.range.option)


class Realsense_Controls(dict):
    def __init__(self, device, presets=()):
        if not device:
            super().__init__()
            return

        if presets:
            # presets: list of (option, value)-tuples
            try:
                device.set_device_options(*zip(*presets))
            except pyrs.RealsenseError as err:
                logger.error('Setting device option presets failed')
                logger.debug('Reason: {}'.format(err))
        controls = {}
        for opt_range, value in device.get_available_options():
            if opt_range.min < opt_range.max:
                controls[opt_range.option] = Control(device, opt_range, value)
        super().__init__(controls)

    def export_presets(self):
        return [(opt, ctrl.value) for opt, ctrl in self.items()]

    def refresh(self):
        for ctrl in self.values():
            ctrl.refresh()


class Realsense_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes pyrs.Device:
    """
    def __init__(self, g_pool, device_id=0,
                 frame_size=(640, 480), frame_rate=30,
                 depth_frame_size=(640, 480), depth_frame_rate=30,
                 align_streams=False, preview_depth=False,
                 device_options=(), record_depth=True):
        super().__init__(g_pool)
        self.color_frame_index = 0
        self.depth_frame_index = 0
        self.device = None
        self.service = pyrs.Service()
        self.align_streams = align_streams
        self.preview_depth = preview_depth
        self.record_depth = record_depth
        self.depth_video_writer = None
        self.controls = None
        self._initialize_device(device_id, frame_size, frame_rate,
                                depth_frame_size, depth_frame_rate, device_options)

    def _initialize_device(self, device_id,
                           color_frame_size, color_fps,
                           depth_frame_size, depth_fps,
                           device_options=()):
        devices = tuple(self.service.get_devices())
        color_frame_size = tuple(color_frame_size)
        depth_frame_size = tuple(depth_frame_size)

        self.streams = [ColorStream(width=1920, height=1080), DepthStream()]
        self.last_color_frame_ts = None
        self.last_depth_frame_ts = None
        self._recent_frame = None
        self._recent_depth_frame = None
        self.deinit_gui()

        if not devices:
            logger.error("Camera failed to initialize. No cameras connected.")
            self.device = None
            self.init_gui()
            return

        if self.device is not None:
            self.device.stop()

        if device_id >= len(devices):
            logger.error("Camera with id {} not found. Initializing default camera.".format(device_id))
            device_id = 0

        # use default streams to filter modes by rs_stream and rs_format
        self._available_modes = self._enumerate_formats(device_id)

        # make sure that given frame sizes and rates are available
        color_modes = self._available_modes[rs_stream.RS_STREAM_COLOR]
        if color_frame_size not in color_modes:
            # automatically select highest resolution
            color_frame_size = sorted(color_modes.keys(), reverse=True)[0]

        if color_fps not in color_modes[color_frame_size]:
            # automatically select highest frame rate
            color_fps = color_modes[color_frame_size][0]

        depth_modes = self._available_modes[rs_stream.RS_STREAM_DEPTH]
        if self.align_streams:
            depth_frame_size = color_frame_size
        else:
            if depth_frame_size not in depth_modes:
                # automatically select highest resolution
                depth_frame_size = sorted(depth_modes.keys(), reverse=True)[0]

        if depth_fps not in depth_modes[depth_frame_size]:
            # automatically select highest frame rate
            depth_fps = depth_modes[depth_frame_size][0]

        colorstream = ColorStream(width=color_frame_size[0],
                                  height=color_frame_size[1],
                                  fps=color_fps, color_format='yuv')
        depthstream = DepthStream(width=depth_frame_size[0],
                                  height=depth_frame_size[1], fps=depth_fps)

        self.streams = [colorstream, depthstream]
        if self.align_streams:
            dacstream = DACStream(width=depth_frame_size[0],
                                  height=depth_frame_size[1], fps=depth_fps)
            dacstream.name = 'depth'  # rename data accessor
            self.streams.append(dacstream)

        # update with correctly initialized streams
        # always initiliazes color + depth, adds rectified/aligned versions as necessary

        self.device = self.service.Device(device_id, streams=self.streams)

        self.controls = Realsense_Controls(self.device, device_options)
        self._intrinsics = load_intrinsics(self.g_pool.user_dir, self.name, self.frame_size)

        self.init_gui()

    def _enumerate_formats(self, device_id):
        '''Enumerate formats into hierachical structure:

        streams:
            resolutions:
                framerates
        '''
        formats = {}
        # only lists modes for native streams (RS_STREAM_COLOR/RS_STREAM_DEPTH)
        for mode in self.service.get_device_modes(device_id):
            if mode.stream in (rs_stream.RS_STREAM_COLOR, rs_stream.RS_STREAM_DEPTH):
                # check if frame size dict is available
                if mode.stream not in formats:
                    formats[mode.stream] = {}
                stream_obj = next((s for s in self.streams if s.stream == mode.stream))
                if mode.format == stream_obj.format:
                    size = mode.width, mode.height
                    # check if framerate list is already available
                    if size not in formats[mode.stream]:
                        formats[mode.stream][size] = []
                    formats[mode.stream][size].append(mode.fps)

        if self.align_streams:
            depth_sizes = formats[rs_stream.RS_STREAM_DEPTH].keys()
            color_sizes = formats[rs_stream.RS_STREAM_COLOR].keys()
            # common_sizes = depth_sizes & color_sizes
            discarded_sizes = depth_sizes ^ color_sizes
            for size in discarded_sizes:
                for sizes in formats.values():
                    if size in sizes:
                        del sizes[size]

        return formats

    def cleanup(self):
        if self.depth_video_writer is not None:
            self.stop_depth_recording()
        if self.device is not None:
            self.device.stop()
        self.service.stop()
        super().cleanup()

    def get_init_dict(self):
        return {'device_id': self.device.device_id if self.device is not None else 0,
                'frame_size': self.frame_size,
                'frame_rate': self.frame_rate,
                'depth_frame_size': self.depth_frame_size,
                'depth_frame_rate': self.depth_frame_rate,
                'preview_depth': self.preview_depth,
                'record_depth': self.record_depth,
                'align_streams': self.align_streams,
                'device_options': self.controls.export_presets() if self.controls is not None else ()}

    def get_frames(self):
        if self.device:
            self.device.wait_for_frames()
            current_time = self.g_pool.get_timestamp()

            last_color_frame_ts = self.device.get_frame_timestamp(self.streams[0].stream)
            if self.last_color_frame_ts != last_color_frame_ts:
                self.last_color_frame_ts = last_color_frame_ts
                color = ColorFrame(self.device)
                color.timestamp = current_time
                color.index = self.color_frame_index
                self.color_frame_index += 1
            else:
                color = None

            last_depth_frame_ts = self.device.get_frame_timestamp(self.streams[1].stream)
            if self.last_depth_frame_ts != last_depth_frame_ts:
                self.last_depth_frame_ts = last_depth_frame_ts
                depth = DepthFrame(self.device)
                depth.timestamp = current_time
                depth.index = self.depth_frame_index
                self.depth_frame_index += 1
            else:
                depth = None

            return color, depth
        return None, None

    def recent_events(self, events):
        if not self.online:
            time.sleep(.05)
            return

        try:
            color_frame, depth_frame = self.get_frames()
        except (pyrs.RealsenseError, TimeoutError) as err:
            self._recent_frame = None
            self._recent_depth_frame = None
            self.restart_device()
        else:
            if color_frame and depth_frame:
                self._recent_frame = color_frame
                events['frame'] = color_frame

            if depth_frame:
                self._recent_depth_frame = depth_frame
                events['depth_frame'] = depth_frame

                if self.depth_video_writer is not None:
                    self.depth_video_writer.write_video_frame(depth_frame)

    def init_gui(self):
        from pyglui import ui
        ui_elements = []

        # avoid duplicated elements since _initialize_device() calls init_gui as well
        self.deinit_gui()

        if self.device is None:
            ui_elements.append(ui.Info_Text('Capture initialization failed.'))
            self.g_pool.capture_source_menu.extend(ui_elements)
            return

        def align_and_restart(val):
            self.align_streams = val
            self.restart_device()

        ui_elements.append(ui.Switch('record_depth', self, label='Record Depth Stream'))
        ui_elements.append(ui.Switch('preview_depth', self, label='Preview Depth'))
        ui_elements.append(ui.Switch('align_streams', self, label='Align Streams',
                                     setter=align_and_restart))

        color_sizes = sorted(self._available_modes[rs_stream.RS_STREAM_COLOR], reverse=True)
        ui_elements.append(ui.Selector(
            'frame_size', self,
            # setter=,
            selection=color_sizes,
            label= 'Resolution' if self.align_streams else 'Color Resolution'
        ))

        def color_fps_getter():
            avail_fps = self._available_modes[rs_stream.RS_STREAM_COLOR][self.frame_size]
            return avail_fps, [str(fps) for fps in avail_fps]
        ui_elements.append(ui.Selector(
            'frame_rate', self,
            # setter=,
            selection_getter=color_fps_getter,
            label='Color Frame Rate'
        ))

        if not self.align_streams:
            depth_sizes = sorted(self._available_modes[rs_stream.RS_STREAM_DEPTH], reverse=True)
            ui_elements.append(ui.Selector(
                'depth_frame_size', self,
                # setter=,
                selection=depth_sizes,
                label='Depth Resolution'
            ))

        def depth_fps_getter():
            avail_fps = self._available_modes[rs_stream.RS_STREAM_DEPTH][self.depth_frame_size]
            return avail_fps, [str(fps) for fps in avail_fps]
        ui_elements.append(ui.Selector(
            'depth_frame_rate', self,
            selection_getter=depth_fps_getter,
            label='Depth Frame Rate'
        ))

        def reset_options():
            if self.device:
                try:
                    self.device.reset_device_options_to_default(self.controls.keys())
                except pyrs.RealsenseError as err:
                    logger.info('Resetting some device options failed')
                    logger.debug('Reason: {}'.format(err))
                finally:
                    self.controls.refresh()

        sensor_control = ui.Growing_Menu(label='Sensor Settings')
        sensor_control.append(ui.Button('Reset device options to default', reset_options))
        for ctrl in sorted(self.controls.values(), key=lambda x: x.range.option):
            # sensor_control.append(ui.Info_Text(ctrl.description))
            if ctrl.range.min == 0.0 and ctrl.range.max == 1.0 and ctrl.range.step == 1.0:
                sensor_control.append(ui.Switch('value', ctrl, label=ctrl.label,
                                                off_val=0.0, on_val=1.0))
            else:
                sensor_control.append(ui.Slider('value', ctrl,
                                                label=ctrl.label,
                                                min=ctrl.range.min,
                                                max=ctrl.range.max,
                                                step=ctrl.range.step))
        ui_elements.append(sensor_control)
        self.g_pool.capture_source_menu.extend(ui_elements)

    def gl_display(self):
        if self.preview_depth and self._recent_depth_frame is not None:
            self.g_pool.image_tex.update_from_ndarray(self._recent_depth_frame.bgr)
            gl_utils.glFlush()
        elif not self.preview_depth and self._recent_frame is not None:
            self.g_pool.image_tex.update_from_yuv_buffer(self._recent_frame.yuv_buffer,self._recent_frame.width,self._recent_frame.height)
            gl_utils.glFlush()

        gl_utils.make_coord_system_norm_based()
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based((self.frame_size[1], self.frame_size[0], 3))

    def restart_device(self, device_id=None, color_frame_size=None, color_fps=None,
                       depth_frame_size=None, depth_fps=None, device_options=None):
        if device_id is None:
            device_id = self.device.device_id
        if color_frame_size is None:
            color_frame_size = self.frame_size
        if color_fps is None:
            color_fps = self.frame_rate
        if depth_frame_size is None:
            depth_frame_size = self.depth_frame_size
        if depth_fps is None:
            depth_fps = self.depth_frame_rate
        if device_options is None:
            device_options = self.controls.export_presets()
        self.service.stop()
        self.service.start()
        self.notify_all({'subject': 'realsense_source.restart',
                         'device_id': device_id,
                         'color_frame_size': color_frame_size,
                         'color_fps': color_fps,
                         'depth_frame_size': depth_frame_size,
                         'depth_fps': depth_fps,
                         'device_options': device_options})

    def on_notify(self, notification):
        if notification['subject'] == 'realsense_source.restart':
            kwargs = notification.copy()
            del kwargs['subject']
            self._initialize_device(**kwargs)
        elif notification['subject'] == 'recording.started':
            self.start_depth_recording(notification['rec_path'])
        elif notification['subject'] == 'recording.stopped':
            self.stop_depth_recording()

    def start_depth_recording(self, rec_loc):
        if not self.record_depth:
            return

        if self.depth_video_writer is not None:
            logger.warning('Depth video recording has been started already')
            return

        video_path = os.path.join(rec_loc, 'depth.mp4')
        self.depth_video_writer = AV_Writer(video_path, fps=self.depth_frame_rate, use_timestamps=True)

    def stop_depth_recording(self):
        if self.depth_video_writer is None:
            logger.warning('Depth video recording was not running')
            return

        self.depth_video_writer.close()
        self.depth_video_writer = None

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def frame_size(self):
        stream = self.streams[0]
        return stream.width, stream.height

    @frame_size.setter
    def frame_size(self, new_size):
        if self.device is not None and new_size != self.frame_size:
            self.restart_device(color_frame_size=new_size)

    @property
    def frame_rate(self):
        return self.streams[0].fps

    @frame_rate.setter
    def frame_rate(self, new_rate):
        if self.device is not None and new_rate != self.frame_rate:
            self.restart_device(color_fps=new_rate)

    @property
    def depth_frame_size(self):
        stream = self.streams[1]
        return stream.width, stream.height

    @depth_frame_size.setter
    def depth_frame_size(self, new_size):
        if self.device is not None and new_size != self.depth_frame_size:
            self.restart_device(depth_frame_size=new_size)

    @property
    def depth_frame_rate(self):
        return self.streams[1].fps

    @depth_frame_rate.setter
    def depth_frame_rate(self, new_rate):
        if self.device is not None and new_rate != self.depth_frame_rate:
            self.restart_device(depth_fps=new_rate)

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return self.device and self.device.is_streaming()

    @property
    def name(self):
        # not the same as `if self.device:`!
        if self.device is not None:
            return self.device.name
        else:
            return "Ghost capture"


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
            if source_uid is None:
                return

            # with pyrs.Service() as service:
            #     if not service.is_device_streaming(source_uid):
            #         logger.error("The selected camera is already in use or blocked.")
            #         return
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
