'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

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
from glfw import glfwGetCursorPos, glfwGetWindowSize, glfwGetCurrentContext

from plugin import Visualizer_Plugin_Base
from player_methods import transparent_image_overlay
from methods import normalize, denormalize
from video_capture import EndofVideoFileError, FileSeekError, FileCaptureError, File_Source

# logging
import logging
logger = logging.getLogger(__name__)


class Empty(object):
    pass


def get_past_timestamp(idx, timestamps):
    """
    recursive function to find the most recent valid timestamp in the past
    """
    if idx == 0:
        # if at the beginning, we can't go back in time.
        return get_future_timestamp(idx, timestamps)
    if timestamps[idx]:
        res = timestamps[idx][-1]
        return res
    else:
        return get_past_timestamp(idx-1, timestamps)


def get_future_timestamp(idx, timestamps):
    """
    recursive function to find most recent valid timestamp in the future
    """
    if idx == len(timestamps)-1:
        # if at the end, we can't go further into the future.
        return get_past_timestamp(idx, timestamps)
    elif timestamps[idx]:
        return timestamps[idx][0]
    else:
        idx = min(len(timestamps), idx+1)
        return get_future_timestamp(idx, timestamps)


def get_nearest_timestamp(past_timestamp, future_timestamp, world_timestamp):
    dt_past = abs(past_timestamp-world_timestamp)
    dt_future = abs(future_timestamp-world_timestamp)  # abs prob not necessary here, but just for sanity
    if dt_past < dt_future:
        return past_timestamp
    else:
        return future_timestamp


def correlate_eye_world(eye_timestamps, world_timestamps):
    """
    This function takes a list of eye timestamps and world timestamps
    and correlates one eye frame per world frame
    Returns a mapping that correlates a single eye frame index with each world frame index.
    Up and downsampling is used to achieve this mapping.
    """
    # return framewise mapping as a list
    e_ts = eye_timestamps
    w_ts = list(world_timestamps)
    eye_frames_by_timestamp = dict(zip(e_ts, range(len(e_ts))))

    eye_timestamps_by_world_index = [[] for i in world_timestamps]

    frame_idx = 0
    try:
        current_e_ts = e_ts.pop(0)
    except IndexError:
        # logger.warning("No eye timestamps at all in the section.")
        return eye_timestamps_by_world_index

    while e_ts:
        # if the current eye timestamp is before the mean of the current
        # world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = (w_ts[frame_idx]+w_ts[frame_idx+1]) / 2.
        except IndexError:
            break
        if current_e_ts <= t_between_frames:
            eye_timestamps_by_world_index[frame_idx].append(current_e_ts)
            current_e_ts = e_ts.pop(0)
        else:
            frame_idx += 1

    idx = 0
    eye_world_frame_map = []
    # some entiries in the `eye_timestamps_by_world_index` might be empty -- no correlated eye timestamp
    # so we will either show the previous frame or next frame - whichever is temporally closest
    for candidate, world_ts in zip(eye_timestamps_by_world_index, w_ts):
        # if there is no candidate, then assign it to the closest timestamp
        if not candidate:
            # get most recent timestamp, either in the past or future
            e_past_ts = get_past_timestamp(idx, eye_timestamps_by_world_index)
            e_future_ts = get_future_timestamp(idx, eye_timestamps_by_world_index)
            eye_world_frame_map.append(eye_frames_by_timestamp[get_nearest_timestamp(e_past_ts, e_future_ts, world_ts)])
        else:
            # TODO - if there is a list of len > 1 - then we should check which is the temporally closest timestamp
            eye_world_frame_map.append(eye_frames_by_timestamp[eye_timestamps_by_world_index[idx][-1]])
        idx += 1

    return eye_world_frame_map


class Eye_Wrapper(object):
    def __init__(self, eyeid, pos, hflip=False, vflip=False):
        super().__init__()
        self.eyeid = eyeid
        self.pos = pos
        self.hflip = hflip
        self.vflip = vflip
        self.source = None
        self.eye_world_frame_map = None
        self.current_eye_frame = None
        self.drag_offset = None
        self.menu = None

    def initliaze_video(self, rec_dir, world_timestamps):
        eye_loc = os.path.join(rec_dir, 'eye{}.*'.format(self.eyeid))
        ts_loc = os.path.join(rec_dir, 'eye{}_timestamps.npy'.format(self.eyeid))
        try:
            eye_timestamps = np.load(ts_loc)
            self.source = File_Source(Empty(), source_path=glob(eye_loc)[0], timestamps=eye_timestamps)
            self.current_eye_frame = self.source.get_frame()
        except (FileNotFoundError, IndexError, FileCaptureError):
            logger.warning('Video for eye{} was not found or could not be opened.'.format(self.eyeid))
        else:
            self.eye_world_frame_map = correlate_eye_world(eye_timestamps.tolist(), world_timestamps)
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
                self.source.seek_to_frame(requested_eye_frame_idx)

            try:
                self.current_eye_frame = self.source.get_frame()
            except EndofVideoFileError:
                logger.info("Reached the end of the eye video for eye video {}.".format(self.eyeid))

        # 2. dragging image
        if self.drag_offset is not None:
            pos = glfwGetCursorPos(glfwGetCurrentContext())
            pos = normalize(pos, glfwGetWindowSize(glfwGetCurrentContext()))
            # Position in img pixels
            pos = denormalize(pos, (frame.img.shape[1], frame.img.shape[0]))
            self.pos = int(pos[0] + self.drag_offset[0]), int(pos[1] + self.drag_offset[1])

        # 3. keep in image bounds, do this even when not dragging because the image video_sizes could change.
        video_size = round(self.current_eye_frame.width * scale), round(self.current_eye_frame.height * scale)

        # frame.img.shape[0] is height, frame.img.shape[1] is width of screen
        self.pos = (min(frame.img.shape[1] - video_size[0], max(self.pos[0], 0)),
                    min(frame.img.shape[0] - video_size[1], max(self.pos[1], 0)))

        # 4. vflipping images, converting to greyscale
        eye_gray = self.current_eye_frame.gray
        eyeimage = cv2.resize(eye_gray, (0, 0), fx=scale, fy=scale)
        if self.hflip:
            eyeimage = np.fliplr(eyeimage)
        if self.vflip:
            eyeimage = np.flipud(eyeimage)

        eyeimage = cv2.cvtColor(eyeimage, cv2.COLOR_GRAY2BGR)
        if show_ellipses:
            try:
                pp = next((pp for pp in pupil_positions if pp['id'] == self.eyeid and pp['timestamp'] == self.current_eye_frame.timestamp))
            except StopIteration:
                pass
            else:
                el = pp['ellipse']
                conf = int(pp.get('model_confidence', pp.get('confidence', 0.1)) * 255)
                center = list(map(lambda val: int(scale*val), el['center']))
                el['axes'] = tuple(map(lambda val: int(scale*val/2), el['axes']))
                el['angle'] = int(el['angle'])
                el_points = cv2.ellipse2Poly(tuple(center), el['axes'], el['angle'], 0, 360, 1)
                if self.hflip:
                    el_points = [(video_size[0] - x, y) for x, y in el_points]
                    center[0] = video_size[0] - center[0]
                if self.vflip:
                    el_points = [(x, video_size[1] - y) for x, y in el_points]
                    center[1] = video_size[1] - center[1]

                cv2.polylines(eyeimage, [np.asarray(el_points)], True, (0, 0, 255, conf), thickness=int(np.ceil(2*scale)))
                cv2.circle(eyeimage, tuple(center), int(5*scale), (0, 0, 255, conf), thickness=-1)

        transparent_image_overlay(self.pos, eyeimage, frame.img, alpha)

    def on_click(self, pos, button, action, scale):
        if not self.initialized:
            return False  # click event has not been consumed

        video_size = round(self.current_eye_frame.width * scale), round(self.current_eye_frame.height * scale)

        if (self.pos[0] < pos[0] < self.pos[0] + video_size[0] and
                self.pos[1] < pos[1] < self.pos[1] + video_size[1]):
            self.drag_offset = self.pos[0] - pos[0], self.pos[1] - pos[1]
            return True
        else:
            self.drag_offset = None
            return False


class Vis_Eye_Video_Overlay(Visualizer_Plugin_Base):
    def __init__(self, g_pool, alpha=0.6, eye_scale_factor=.5, show_ellipses=True,
                 eye0_config={'pos': (640, 10)}, eye1_config={'pos': (10, 10)}):
        super().__init__(g_pool)
        self.order = .6
        self.menu = None
        self.alpha = alpha
        self.eye_scale_factor = eye_scale_factor
        self.show_ellipses = show_ellipses
        self.move_around = False

        self.eye0 = Eye_Wrapper(0, **eye0_config)
        self.eye1 = Eye_Wrapper(1, **eye1_config)

        self.eye0.initliaze_video(g_pool.rec_dir, g_pool.timestamps)
        self.eye1.initliaze_video(g_pool.rec_dir, g_pool.timestamps)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Eye Video Overlay')
        self.g_pool.gui.append(self.menu)

        def close():
            self.alive = False

        self.menu.append(ui.Button('Close', close))
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
            if not self.eye1.on_click(pos, button, action, self.eye_scale_factor):
                self.eye0.on_click(pos, button, action, self.eye_scale_factor)
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

    def deinit_gui(self):
        if self.menu is not None:
            self.eye0.remove_eye_menu(self.menu)
            self.eye1.remove_eye_menu(self.menu)
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {'alpha': self.alpha, 'eye_scale_factor': self.eye_scale_factor, 'show_ellipses': self.show_ellipses,
                'eye0_config': self.eye0.config, 'eye1_config': self.eye1.config}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
