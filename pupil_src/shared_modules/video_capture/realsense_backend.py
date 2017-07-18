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

import pyrealsense as pyrs
from pyrealsense.stream import ColorStream, DepthStream
from pyrealsense.constants import rs_stream

from version_utils import VersionFormat
from .base_backend import Base_Source, Base_Manager

import gl_utils
from pyglui import cygl
import numpy as np

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(pyrs.__version__) >= VersionFormat('2.1')

# logging
logging.getLogger('pyrealsense').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ColorFrame(object):
    def __init__(self, device):
        self._rgb = device.color

    @property
    def bgr(self):
        return self._rgb


class DepthFrame(object):
    def __init__(self, device):
        depth = 255 * (device.depth / device.depth.max()) * (65 / 4)

        depth = depth.astype(np.uint8)
        self.mapped_depth = cv2.applyColorMap(depth, cv2.COLORMAP_BONE)

    @property
    def bgr(self):
        return self.mapped_depth


class Realsense_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes pyrs.Device:
    """
    def __init__(self, g_pool, device_id=0,
                 frame_size=(640, 480), frame_rate=30,
                 depth_frame_size=(640, 480), depth_frame_rate=30,
                 align_streams=False, rectify_streams=False,
                 preview_depth=False, device_options={}):
        super().__init__(g_pool)
        self.device = None
        self.service = pyrs.Service()
        self.align_streams = align_streams
        self.rectify_streams = rectify_streams
        self.preview_depth = preview_depth
        self._initialize_device(device_id, tuple(frame_size), frame_rate,
                                tuple(depth_frame_size), depth_frame_rate, device_options)

    def _initialize_device(self, device_id,
                           color_frame_size, color_fps,
                           depth_frame_size, depth_fps,
                           device_options=None):
        devices = tuple(self.service.get_devices())
        if not devices:
            logger.error("Camera failed to initialize. No cameras connected.")
            self.device = None
            return

        if self.device is not None:
            self.device.stop()

        if device_id >= len(devices):
            logger.error("Camera with id {} not found. Initializing default camera.".format(device_id))
            device_id = 0

        # use default streams to filter modes by rs_stream and rs_format
        self.streams = {s.stream: s for s in (ColorStream(), DepthStream())}
        self._available_modes = self._enumerate_formats(device_id)

        # make sure that given frame sizes and rates are available
        color_modes = self._available_modes[rs_stream.RS_STREAM_COLOR]
        if color_frame_size not in color_modes:
            # automatically select highest resolution
            color_frame_size = sorted(color_modes.keys(), reversed=True)[0]

        if color_fps not in color_modes[color_frame_size]:
            # automatically select highest frame rate
            color_fps = color_modes[color_frame_size][0]

        if self.align_streams:
            depth_frame_size = color_frame_size
            depth_fps = color_fps
        else:
            depth_modes = self._available_modes[rs_stream.RS_STREAM_DEPTH]
            if depth_frame_size not in depth_modes:
                # automatically select highest resolution
                depth_frame_size = sorted(depth_modes.keys(), reversed=True)[0]

            if depth_fps not in depth_modes[depth_frame_size]:
                # automatically select highest frame rate
                depth_fps = depth_modes[depth_frame_size][0]

        colorstream = ColorStream(width=color_frame_size[0],
                                  height=color_frame_size[1],
                                  fps=color_fps, use_bgr=True)
        depthstream = DepthStream(width=depth_frame_size[0],
                                  height=depth_frame_size[1], fps=depth_fps)

        # update with correctly initialized streams
        self.streams.update({s.stream: s for s in (colorstream, depthstream)})

        self.device = self.service.Device(device_id, streams=self.streams.values())

        if not device_options:
            pass  # apply device options

    def _enumerate_formats(self, device_id):
        '''Enumerate formats into hierachical structure:

        streams:
            resolutions:
                framerates
        '''
        formats = {}
        # only lists modes for native streams (RS_STREAM_COLOR/RS_STREAM_DEPTH)
        for mode in self.service.get_device_modes(device_id):
            if mode.stream in self.streams:
                # check if frame size dict is available
                if mode.stream not in formats:
                    formats[mode.stream] = {}
                stream_obj = self.streams[mode.stream]
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
        self.service.stop()
        super().cleanup()

    def get_init_dict(self):
        return {'device_id': self.device.device_id,
                'frame_size': self.frame_size,
                # 'device_options': self.device_options,  # shoud enumerate current state
                'frame_rate': self.frame_rate,
                'depth_frame_size': self.depth_frame_size,
                'depth_frame_rate': self.depth_frame_rate,
                'preview_depth': self.preview_depth,
                'align_streams': self.align_streams,
                'rectify_streams': self.rectify_streams}

    def get_frame(self, frame_cls):
        if self.device.poll_for_frame() != 0:
            return frame_cls(self.device)
        else:
            max_fps = max([s.fps for s in self.streams.values()])
            time.sleep(0.5/max_fps)

    def recent_events(self, events):
        try:
            if self.preview_depth:
                frame = self.get_frame(DepthFrame)
            else:
                frame = self.get_frame(ColorFrame)
        except TimeoutError:
            self._recent_frame = None
            # react to timeout
        except pyrs.RealsenseError as err:
            self._recent_frame = None
            # act according to err.function
            # self._restart_logic()
        else:
            if frame:
                frame.timestamp = self.g_pool.get_timestamp()
                self._recent_frame = frame
                events['frame'] = frame
                self._restart_in = 3

    def init_gui(self):
        from pyglui import ui
        ui_elements = []

        if self.device is None:
            ui_elements.append(ui.Info_Text('Capture initialization failed.'))
            self.g_pool.capture_source_menu.extend(ui_elements)
            return

        ui_elements.append(ui.Switch('preview_depth', self, label='Preview Depth'))
        ui_elements.append(ui.Switch('rectify_streams', self, label='Rectify Streams'))
        ui_elements.append(ui.Switch('align_streams', self, label='Align Streams'))

        color_sizes = sorted(self._available_modes[rs_stream.RS_STREAM_COLOR], reverse=True)
        ui_elements.append(ui.Selector(
            'frame_size', self,
            # setter=,
            selection=color_sizes,
            label='Color Resolution'
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
            # setter=,
            selection_getter=depth_fps_getter,
            label='Depth Frame Rate'
        ))

        self.g_pool.capture_source_menu.extend(ui_elements)

    def gl_display(self):
        if self._recent_frame is not None:
            self.g_pool.image_tex.update_from_ndarray(self._recent_frame.bgr)
            gl_utils.glFlush()
        gl_utils.make_coord_system_norm_based()
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based((self.frame_size[1], self.frame_size[0], 3))

    @property
    def frame_size(self):
        stream = self.streams[rs_stream.RS_STREAM_COLOR]
        return stream.width, stream.height

    @frame_size.setter
    def frame_size(self, new_size):
        if self.device is not None and new_size != self.frame_size:
            self._initialize_device(self.device.device_id,
                                    new_size, self.frame_rate,
                                    self.depth_frame_size, self.depth_frame_rate,
                                    device_options=None)

    @property
    def frame_rate(self):
        return self.streams[rs_stream.RS_STREAM_COLOR].fps

    @frame_rate.setter
    def frame_rate(self, new_rate):
        if self.device is not None and new_rate != self.frame_rate:
            self._initialize_device(self.device.device_id,
                                    self.frame_size, new_rate,
                                    self.depth_frame_size, self.depth_frame_rate,
                                    device_options=None)

    @property
    def depth_frame_size(self):
        stream = self.streams[rs_stream.RS_STREAM_DEPTH]
        return stream.width, stream.height

    @depth_frame_size.setter
    def depth_frame_size(self, new_size):
        if self.device is not None and new_size != self.depth_frame_size:
            self._initialize_device(self.device.device_id,
                                    self.frame_size, self.frame_rate,
                                    new_size, self.depth_frame_rate,
                                    device_options=None)

    @property
    def depth_frame_rate(self):
        return self.streams[rs_stream.RS_STREAM_DEPTH].fps

    @depth_frame_rate.setter
    def depth_frame_rate(self, new_rate):
        if self.device is not None and new_rate != self.depth_frame_rate:
            self._initialize_device(self.device.device_id,
                                    self.frame_size, self.frame_rate,
                                    self.depth_frame_size, new_rate,
                                    device_options=None)

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return self.device.is_streaming()

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
