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
from pyrealsense.stream import ColorStream, DepthStream, DACStream, PointStream
from pyrealsense.constants import rs_stream, rs_option, rs_preset
from pyrealsense.extlib import rsutilwrapper

from version_utils import VersionFormat
from .base_backend import Base_Source, Base_Manager
from av_writer import AV_Writer
from camera_models import load_intrinsics

import glfw
import gl_utils
from OpenGL.GL import *
from OpenGL.GLU import *
from pyglui import cygl
import cython_methods
import numpy as np
from ctypes import *

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
            self._bgr = cv2.cvtColor(self._yuv422, cv2.COLOR_YUV2BGR_YUYV)
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
                 frame_size=(1920, 1080), frame_rate=30,
                 depth_frame_size=(640, 480), depth_frame_rate=60,
                 align_streams=False, preview_depth=False,
                 device_options=(), record_depth=True, stream_preset = None):
        super().__init__(g_pool)
        self._intrinsics = None
        self.color_frame_index = 0
        self.depth_frame_index = 0
        self.device = None
        self.service = pyrs.Service()
        self.align_streams = align_streams
        self.preview_depth = preview_depth
        self.record_depth = record_depth
        self.depth_video_writer = None
        self.controls = None
        self.pitch = 0
        self.yaw = 0
        self.mouse_drag = False
        self.last_pos = (0,0)
        self.depth_window = None
        self._needs_restart = False
        self.stream_preset = stream_preset
        self._initialize_device(device_id, frame_size, frame_rate,
                                depth_frame_size, depth_frame_rate, device_options)

    def _initialize_device(self, device_id,
                           color_frame_size, color_fps,
                           depth_frame_size, depth_fps,
                           device_options=()):
        devices = tuple(self.service.get_devices())
        color_frame_size = tuple(color_frame_size)
        depth_frame_size = tuple(depth_frame_size)

        self.streams = [ColorStream(), DepthStream(), PointStream()]
        self.last_color_frame_ts = None
        self.last_depth_frame_ts = None
        self._recent_frame = None
        self._recent_depth_frame = None

        if not devices:
            if not self._needs_restart:
                logger.error("Camera failed to initialize. No cameras connected.")
            self.device = None
            self.update_menu()
            return

        if self.device is not None:
            self.device.stop()  # only call Device.stop() if its context

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
                                  fps=color_fps, color_format='yuv', preset=self.stream_preset)
        depthstream = DepthStream(width=depth_frame_size[0],
                                  height=depth_frame_size[1], fps=depth_fps, preset=self.stream_preset)
        pointstream = PointStream(width=depth_frame_size[0],
                                  height=depth_frame_size[1], fps=depth_fps)

        self.streams = [colorstream, depthstream, pointstream]
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

        self.update_menu()
        self._needs_restart = False

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

    def get_init_dict(self):
        return {'device_id': self.device.device_id if self.device is not None else 0,
                'frame_size': self.frame_size,
                'frame_rate': self.frame_rate,
                'depth_frame_size': self.depth_frame_size,
                'depth_frame_rate': self.depth_frame_rate,
                'preview_depth': self.preview_depth,
                'record_depth': self.record_depth,
                'align_streams': self.align_streams,
                'device_options': self.controls.export_presets() if self.controls is not None else (),
                'stream_preset': self.stream_preset}

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
        if self._needs_restart:
            self.restart_device()
            time.sleep(0.05)
        elif not self.online:
            time.sleep(.05)
            return

        try:
            color_frame, depth_frame = self.get_frames()
        except (pyrs.RealsenseError, TimeoutError) as err:
            logger.warning("Realsense failed to provide frames. Attempting to reinit.")
            self._recent_frame = None
            self._recent_depth_frame = None
            self._needs_restart = True
        else:
            if color_frame and depth_frame:
                self._recent_frame = color_frame
                events['frame'] = color_frame

            if depth_frame:
                self._recent_depth_frame = depth_frame
                events['depth_frame'] = depth_frame

                if self.depth_video_writer is not None:
                    self.depth_video_writer.write_video_frame(depth_frame)

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Local USB Video Source"
        self.update_menu()

    def update_menu(self):
        try:
            del self.menu[:]
        except AttributeError:
            return

        from pyglui import ui

        if self.device is None:
            self.menu.append(ui.Info_Text('Capture initialization failed.'))
            return

        def align_and_restart(val):
            self.align_streams = val
            self.restart_device()

        self.menu.append(ui.Switch('record_depth', self, label='Record Depth Stream'))
        self.menu.append(ui.Switch('preview_depth', self, label='Preview Depth'))
        self.menu.append(ui.Switch('align_streams', self, label='Align Streams',
                                   setter=align_and_restart))
        def toggle_depth_display():
            def on_depth_mouse_button(window, button, action, mods):
                if button == glfw.GLFW_MOUSE_BUTTON_LEFT and action == glfw.GLFW_PRESS:
                   self.mouse_drag = True
                if button == glfw.GLFW_MOUSE_BUTTON_LEFT and action == glfw.GLFW_RELEASE:
                   self.mouse_drag = False

            if self.depth_window is None:
                self.pitch = 0
                self.yaw = 0

                win_size = glfw.glfwGetWindowSize(self.g_pool.main_window)
                self.depth_window = glfw.glfwCreateWindow(win_size[0], win_size[1], "3D Point Cloud")
                glfw.glfwSetMouseButtonCallback(self.depth_window, on_depth_mouse_button)
                active_window = glfw.glfwGetCurrentContext()
                glfw.glfwMakeContextCurrent(self.depth_window)
                gl_utils.basic_gl_setup()
                gl_utils.make_coord_system_norm_based()

                # refresh speed settings
                glfw.glfwSwapInterval(0)

                glfw.glfwMakeContextCurrent(active_window)


        native_presets = [('None', None), ('Best Quality', rs_preset.RS_PRESET_BEST_QUALITY),
                          ('Largest image', rs_preset.RS_PRESET_LARGEST_IMAGE),
                          ('Highest framerate', rs_preset.RS_PRESET_HIGHEST_FRAMERATE)]

        def set_stream_preset(val):
            if self.stream_preset != val:
                self.stream_preset = val
                self.restart_device()
        self.menu.append(ui.Selector(
            'stream_preset', self,
            setter=set_stream_preset,
            labels = [preset[0] for preset in native_presets],
            selection=[preset[1] for preset in native_presets],
            label= 'Stream preset'
        ))
        color_sizes = sorted(self._available_modes[rs_stream.RS_STREAM_COLOR], reverse=True)
        selector = ui.Selector(
            'frame_size', self,
            # setter=,
            selection=color_sizes,
            label= 'Resolution' if self.align_streams else 'Color Resolution')
        selector.read_only = self.stream_preset is not None
        self.menu.append(selector)

        def color_fps_getter():
            avail_fps = [fps for fps in self._available_modes[rs_stream.RS_STREAM_COLOR][self.frame_size] if self.depth_frame_rate % fps == 0]
            return avail_fps, [str(fps) for fps in avail_fps]
        selector = ui.Selector(
            'frame_rate', self,
            # setter=,
            selection_getter=color_fps_getter,
            label='Color Frame Rate',
        )
        selector.read_only = self.stream_preset is not None
        self.menu.append(selector)

        if not self.align_streams:
            depth_sizes = sorted(self._available_modes[rs_stream.RS_STREAM_DEPTH], reverse=True)
            selector = ui.Selector(
                'depth_frame_size', self,
                # setter=,
                selection=depth_sizes,
                label='Depth Resolution',
            )
            selector.read_only = self.stream_preset is not None
            self.menu.append(selector)

        def depth_fps_getter():
            avail_fps = [fps for fps in self._available_modes[rs_stream.RS_STREAM_DEPTH][self.depth_frame_size] if fps % self.frame_rate == 0]
            return avail_fps, [str(fps) for fps in avail_fps]
        selector = ui.Selector(
            'depth_frame_rate', self,
            selection_getter=depth_fps_getter,
            label='Depth Frame Rate',
        )
        selector.read_only = self.stream_preset is not None
        self.menu.append(selector)

        def reset_options():
            if self.device:
                try:
                    self.device.reset_device_options_to_default(self.controls.keys())
                except pyrs.RealsenseError as err:
                    logger.info('Resetting some device options failed')
                    logger.debug('Reason: {}'.format(err))
                finally:
                    self.controls.refresh()

        self.menu.append(ui.Button('Point Cloud Window', toggle_depth_display))
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
        self.menu.append(sensor_control)

    def gl_display(self):
        from math import floor
        if self.depth_window is not None and glfw.glfwWindowShouldClose(self.depth_window):
            glfw.glfwDestroyWindow(self.depth_window)
            self.depth_window = None

        if self.depth_window is not None and self._recent_depth_frame is not None:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self.depth_window)

            win_size = glfw.glfwGetFramebufferSize(self.depth_window)
            gl_utils.adjust_gl_view(win_size[0], win_size[1])
            pos = glfw.glfwGetCursorPos(self.depth_window)
            if self.mouse_drag:
                self.pitch = np.clip(self.pitch + (pos[1] - self.last_pos[1]), -80, 80)
                self.yaw = np.clip(self.yaw - (pos[0] - self.last_pos[0]), -120, 120)
            self.last_pos = pos

            glClearColor(0,0,0,0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(60, win_size[0]/win_size[1] , 0.01, 20.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(0,0,0, 0,0,1, 0,-1,0)
            glTranslatef(0,0,0.5)
            glRotated(self.pitch, 1, 0, 0)
            glRotated(self.yaw, 0, 1, 0)
            glTranslatef(0,0,-0.5)

            #glPointSize(2)
            glEnable(GL_DEPTH_TEST);
            extrinsics = self.device.get_device_extrinsics(rs_stream.RS_STREAM_DEPTH, rs_stream.RS_STREAM_COLOR)
            depth_frame = self._recent_depth_frame
            color_frame = self._recent_frame
            depth_scale = self.device.depth_scale

            glEnableClientState( GL_VERTEX_ARRAY )

            pointcloud = self.device.pointcloud
            glVertexPointer(3,GL_FLOAT,0,pointcloud)
            glEnableClientState(GL_COLOR_ARRAY);
            depth_to_color = np.zeros(depth_frame.height * depth_frame.width * 3, np.uint8)
            rsutilwrapper.project_pointcloud_to_pixel(depth_to_color, self.device.depth_intrinsics, self.device.color_intrinsics, extrinsics, pointcloud, self._recent_frame.bgr)
            glColorPointer(3, GL_UNSIGNED_BYTE,0, depth_to_color)
            glDrawArrays (GL_POINTS, 0, depth_frame.width * depth_frame.height)
            gl_utils.glFlush()
            glDisable(GL_DEPTH_TEST)
            # gl_utils.make_coord_system_norm_based()
            glfw.glfwSwapBuffers(self.depth_window)
            glfw.glfwMakeContextCurrent(active_window)

        if self.preview_depth and self._recent_depth_frame is not None:
            self.g_pool.image_tex.update_from_ndarray(self._recent_depth_frame.bgr)
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw()
        elif self._recent_frame is not None:
            self.g_pool.image_tex.update_from_yuv_buffer(self._recent_frame.yuv_buffer,
                                                         self._recent_frame.width,
                                                         self._recent_frame.height)
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw()

        if not self.online:
            super().gl_display()

        gl_utils.make_coord_system_pixel_based((self.frame_size[1], self.frame_size[0], 3))

    def restart_device(self, device_id=None, color_frame_size=None, color_fps=None,
                       depth_frame_size=None, depth_fps=None, device_options=None):
        if device_id is None:
            if self.device is not None:
                device_id = self.device.device_id
            else:
               device_id = 0
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
        if self.device is not None:
            self.device.stop()
            self.device = None
        self.service.stop()
        self.service.start()
        self.notify_all({'subject': 'realsense_source.restart',
                         'device_id': device_id,
                         'color_frame_size': color_frame_size,
                         'color_fps': color_fps,
                         'depth_frame_size': depth_frame_size,
                         'depth_fps': depth_fps,
                         'device_options': device_options})

    def on_click(self, pos, button, action):
        if button == glfw.GLFW_MOUSE_BUTTON_LEFT and action == glfw.GLFW_PRESS:
            self.mouse_drag = True
        if button == glfw.GLFW_MOUSE_BUTTON_LEFT and action == glfw.GLFW_RELEASE:
            self.mouse_drag = False

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

    def init_ui(self):
        self.add_menu()
        from pyglui import ui
        self.menu.append(ui.Info_Text('Intel RealSense 3D sources'))

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

        self.menu.append(ui.Selector(
            'selected_source',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Activate source'
        ))

    def deinit_ui(self):
        self.remove_menu()
