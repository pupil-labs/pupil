'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
from glob import glob

import cv2
import numpy as np
from pyglui import ui
from glfw import glfwGetCursorPos, glfwGetFramebufferSize, glfwGetWindowSize, glfwGetCurrentContext, getHDPIFactor

from plugin import Visualizer_Plugin_Base
from player_methods import transparent_image_overlay
from methods import normalize, denormalize
from video_capture import EndofVideoError, File_Source

# logging
import logging
logger = logging.getLogger(__name__)


class Empty(object):
    pass


def correlate_eye_world(eye_timestamps, world_timestamps):
    """
    This function takes a list of eye timestamps and world timestamps
    and correlates one eye frame per world frame
    Returns a mapping that correlates a single eye frame index with each world frame index.
    Up and downsampling is used to achieve this mapping.
    """
    correlation = np.searchsorted(eye_timestamps, world_timestamps)
    correlation[correlation >= eye_timestamps.size] = eye_timestamps.size - 1
    return correlation


class Eye_Wrapper(object):
    def __init__(self, g_pool, eyeid, pos, hdpi_fac=1., hflip=False, vflip=False):
        super().__init__()
        self.g_pool = g_pool
        self.eyeid = eyeid
        self.pos = pos
        self.hflip = hflip
        self.vflip = vflip
        self.source = None
        self.eye_world_frame_map = None
        self.current_eye_frame = None
        self.drag_offset = None
        self.menu = None
        self.hdpi_fac = hdpi_fac

    def initliaze_video(self, rec_dir, world_timestamps):
        eye_loc = os.path.join(rec_dir, 'eye{}.*'.format(self.eyeid))
        try:
            self.source = File_Source(Empty(), source_path=glob(eye_loc)[0], timing=None)
            self.current_eye_frame = self.source.get_frame()
        except (FileNotFoundError, IndexError):
            logger.warning('Video for eye{} was not found or could not be opened.'.format(self.eyeid))
        else:
            self.eye_world_frame_map = correlate_eye_world(self.source.timestamps, world_timestamps)
            if self.menu is not None:
                self.menu.read_only = False

    def add_eye_menu(self, parent):
        self.menu = ui.Growing_Menu('Eye {}'.format(self.eyeid))
        parent.append(self.menu)
        self.menu.append(ui.Switch('hflip', self, label='Horizontal flip'))
        self.menu.append(ui.Switch('vflip', self, label='Vertical flip'))
        self.menu.read_only = not self.initialized

    def remove_eye_menu(self, parent):
        parent.remove(self.menu)
        self.menu = None

    def deinitliaze_video(self):
        self.source = None
        self.eye_world_frame_map = None
        self.current_eye_frame = None
        if self.menu is not None:
            self.menu.read_only = True

    @property
    def initialized(self):
        return self.source is not None

    @property
    def config(self):
        return {'pos': self.pos, 'hflip': self.hflip, 'vflip': self.vflip}

    def visualize(self, frame, alpha, scale, show_ellipses, pupil_positions):
        if not self.initialized:
            return

        requested_eye_frame_idx = self.eye_world_frame_map[frame.index]
        # 1. do we need a new frame?
        if requested_eye_frame_idx != self.current_eye_frame.index:
            if requested_eye_frame_idx == self.source.get_frame_index() + 2:
                # if we just need to seek by one frame, its faster to just read one and and throw it away.
                self.source.get_frame()
            if requested_eye_frame_idx != self.source.get_frame_index() + 1:
                self.source.seek_to_frame(int(requested_eye_frame_idx))

            try:
                self.current_eye_frame = self.source.get_frame()
            except EndofVideoError:
                logger.info("Reached the end of the eye video for eye video {}.".format(self.eyeid))

        # 2. dragging image
        if self.drag_offset is not None:
            x, y = glfwGetCursorPos(glfwGetCurrentContext())
            pos = x * self.hdpi_fac, y * self.hdpi_fac
            pos = normalize(pos, self.g_pool.camera_render_size)
            # Position in img pixels
            pos = denormalize(pos, (frame.img.shape[1], frame.img.shape[0]))
            self.pos = int(pos[0] + self.drag_offset[0]), int(pos[1] + self.drag_offset[1])

        # 3. keep in image bounds, do this even when not dragging because the image video_sizes could change.
        video_size = round(self.current_eye_frame.width * scale), round(self.current_eye_frame.height * scale)

        # frame.img.shape[0] is height, frame.img.shape[1] is width of screen
        self.pos = (min(frame.img.shape[1] - video_size[0], max(self.pos[0], 0)),
                    min(frame.img.shape[0] - video_size[1], max(self.pos[1], 0)))


        # 4. vflipping images, converting to greyscale
        eyeimage = self.current_eye_frame.gray
        eyeimage = cv2.cvtColor(eyeimage, cv2.COLOR_GRAY2BGR)

        if show_ellipses:
            try:
                pp = next((pp for pp in pupil_positions if pp['id'] == self.eyeid and pp['timestamp'] == self.current_eye_frame.timestamp))
            except StopIteration:
                pass
            else:
                el = pp['ellipse']
                conf = int(pp.get('model_confidence', pp.get('confidence', 0.1)) * 255)
                el_points = getEllipsePts((el['center'], el["axes"], el['angle']))
                cv2.polylines(eyeimage, [np.asarray(el_points,dtype='i')], True, (0, 0, 255, conf), thickness=1)
                cv2.circle(eyeimage,(int(el['center'][0]),int(el['center'][1])), 5, (0, 0, 255, conf), thickness=-1)


        #flip and scale
        eyeimage = cv2.resize(eyeimage, (0, 0), fx=scale, fy=scale)
        if self.hflip:
            eyeimage = np.fliplr(eyeimage)
        if self.vflip:
            eyeimage = np.flipud(eyeimage)

        transparent_image_overlay(self.pos, eyeimage, frame.img, alpha)

    def on_click(self, pos, button, action, hdpi_fac, eye_scale):
        self.hdpi_fac = hdpi_fac
        if not self.initialized:
            return False  # click event has not been consumed

        video_size = round(self.current_eye_frame.width * eye_scale), round(self.current_eye_frame.height * eye_scale)

        if (self.pos[0] < pos[0] < self.pos[0] + video_size[0] and
                self.pos[1] < pos[1] < self.pos[1] + video_size[1]):
            self.drag_offset = self.pos[0] - pos[0], self.pos[1] - pos[1]
            return True
        else:
            self.drag_offset = None
            return False


def getEllipsePts(e, num_pts=10):
    c1 = e[0][0]
    c2 = e[0][1]
    a = e[1][0]
    b = e[1][1]
    angle = e[2]

    steps = np.linspace(0, 2 * np.pi, num=num_pts, endpoint=False)
    rot = cv2.getRotationMatrix2D((0, 0), -angle, 1)

    pts1 = a / 2.0 * np.cos(steps)
    pts2 = b / 2.0 * np.sin(steps)
    pts = np.column_stack((pts1, pts2, np.ones(pts1.shape[0])))

    pts_rot = np.matmul(rot, pts.T)
    pts_rot = pts_rot.T

    pts_rot[:, 0] += c1
    pts_rot[:, 1] += c2

    return pts_rot


class Vis_Eye_Video_Overlay(Visualizer_Plugin_Base):
    icon_chr = chr(0xec02)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, alpha=0.6, eye_scale_factor=.5, show_ellipses=True,
                 eye0_config={'pos': (640, 10)}, eye1_config={'pos': (10, 10)}):
        super().__init__(g_pool)
        self.order = .6
        self.menu = None
        self.alpha = alpha
        self.eye_scale_factor = eye_scale_factor
        self.show_ellipses = show_ellipses
        self.move_around = False

        if self.g_pool.app != 'exporter':
            window = g_pool.main_window
            self.hdpi_factor = getHDPIFactor(window)
        else:
            self.hdpi_factor = 1.

        self.eye0 = Eye_Wrapper(g_pool, 0, hdpi_fac=self.hdpi_factor, **eye0_config)
        self.eye1 = Eye_Wrapper(g_pool, 1, hdpi_fac=self.hdpi_factor, **eye1_config)

        self.eye0.initliaze_video(g_pool.rec_dir, g_pool.timestamps)
        self.eye1.initliaze_video(g_pool.rec_dir, g_pool.timestamps)

    def init_ui(self):
        self.add_menu()
        # initialize the menu
        self.menu.label = 'Eye Video Overlay'
        self.menu.append(ui.Info_Text('Show the eye video overlaid on top of the world video. Eye 0 is usually the right eye.'))
        self.menu.append(ui.Slider('alpha', self, min=0.0, step=0.05, max=1.0, label='Opacity'))
        self.menu.append(ui.Slider('eye_scale_factor', self, min=0.2, step=0.1, max=1.0, label='Video Scale'))
        self.menu.append(ui.Switch('show_ellipses', self, label="Visualize Ellipses"))
        self.menu.append(ui.Switch('move_around', self, label="Move Overlay"))

        def add_eye_switch(wrapper):
            def display_eye(should_show):
                if should_show:
                    wrapper.initliaze_video(self.g_pool.rec_dir, self.g_pool.timestamps)
                else:
                    wrapper.deinitliaze_video()

            self.menu.append(ui.Switch('initialized', wrapper,
                                       label="Show Eye {}".format(wrapper.eyeid),
                                       setter=display_eye))

        add_eye_switch(self.eye0)
        add_eye_switch(self.eye1)

        self.eye0.add_eye_menu(self.menu)
        self.eye1.add_eye_menu(self.menu)

    def on_click(self, pos, button, action):
        # eye1 is drawn above eye0. Therefore eye1 gets the priority during click event handling
        # wrapper.on_click returns bool indicating the consumption of the click event
        if self.move_around and action == 1:
            if not self.eye1.on_click(pos, button, action, self.hdpi_factor, self.eye_scale_factor):
                self.eye0.on_click(pos, button, action, self.hdpi_factor, self.eye_scale_factor)
        else:
            self.eye0.drag_offset = None
            self.eye1.drag_offset = None

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        self.eye0.visualize(frame, self.alpha, self.eye_scale_factor,
                            self.show_ellipses, events['pupil_positions'])
        self.eye1.visualize(frame, self.alpha, self.eye_scale_factor,
                            self.show_ellipses, events['pupil_positions'])

    def deinit_ui(self):
        self.remove_menu()

    def get_init_dict(self):
        return {'alpha': self.alpha, 'eye_scale_factor': self.eye_scale_factor, 'show_ellipses': self.show_ellipses,
                'eye0_config': self.eye0.config, 'eye1_config': self.eye1.config}

    def on_window_resize(self, window, camera_render_width, camera_render_height):
        self.hdpi_factor = float(glfwGetFramebufferSize(window)[0] / glfwGetWindowSize(window)[0])

