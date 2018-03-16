'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from .base_backend import Base_Source, Playback_Source, Base_Manager, EndofVideoError

import os
import cv2
import numpy as np
from time import time, sleep
from pyglui import ui
from camera_models import Dummy_Camera
from file_methods import load_object

# logging
import logging
logger = logging.getLogger(__name__)


class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp, img, index):
        self.timestamp = timestamp
        self._img = img
        self.bgr = img
        self.height, self.width, _ = img.shape
        self._gray = None
        self.index = index
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    @property
    def img(self):
        return self._img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        return self._gray

    def copy(self):
        return Frame(self.timestamp, self._img.copy(), self.index)


class Fake_Source(Playback_Source, Base_Source):
    """Simple source which shows random, static image.

    It is used as falback in case the original source fails. `preferred_source`
    contains the necessary information to recover to the original source if
    it becomes accessible again.

    Attributes:
        current_frame_idx (int): Sequence counter
        frame_rate (int)
        frame_size (tuple)
    """
    def __init__(self, g_pool, source_path=None, frame_size=None,
                 frame_rate=None, name='Fake Source', *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        if self.timing == 'external':
            self.recent_events = self.recent_events_external_timing
        else:
            self.recent_events = self.recent_events_own_timing

        if source_path:
            meta = load_object(source_path)
            frame_size = meta['frame_size']
            frame_rate = meta['frame_rate']
            self.timestamps = np.load(os.path.splitext(source_path)[0] + '_timestamps.npy')
        else:
            self.timestamps = None

        self.fps = frame_rate
        self._name = name
        self.make_img(tuple(frame_size))
        self.source_path = source_path
        self.current_frame_idx = 0
        self.target_frame_idx = 0

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Static Image Source"

        text = ui.Info_Text("This plugin displays a static image.")
        self.menu.append(text)

        self.menu.append(ui.Text_Input('frame_size', label='Frame size',
                                       setter=lambda x: None,
                                       getter=lambda: '{} x {}'.format(*self.frame_size)))

        self.menu.append(ui.Text_Input('frame_rate', label='Frame rate',
                                       setter=lambda x: None,
                                       getter=lambda: '{:.0f} FPS'.format(self.frame_rate)))

        if self.g_pool.app == 'player':
            # get_frame_count() is not constant in Capture and would trigger
            # a redraw on each frame
            self.menu.append(ui.Text_Input('frame_num', label='Number of frames',
                                           setter=lambda x: None,
                                           getter=lambda: self.get_frame_count()))

    def deinit_ui(self):
        self.remove_menu()

    def make_img(self, size):
        # Generate Pupil Labs colored gradient
        self._img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._img[:, :, 0] += np.linspace(91, 157, self.frame_size[0], dtype=np.uint8)
        self._img[:, :, 1] += np.linspace(165, 161, self.frame_size[0], dtype=np.uint8)
        self._img[:, :, 2] += np.linspace(35, 112, self.frame_size[0], dtype=np.uint8)

        self._intrinsics = Dummy_Camera(size, self.name)

    def recent_events_external_timing(self, events):
        try:
            last_index = self._recent_frame.index
        except AttributeError:
            # called once on start when self._recent_frame is None
            last_index = -1

        frame = None
        pbt = self.g_pool.seek_control.current_playback_time
        ts_idx = self.g_pool.seek_control.ts_idx_from_playback_time(pbt)
        if ts_idx == last_index:
            frame = self._recent_frame.copy()
            if self.play and ts_idx == self.get_frame_count() - 1:
                logger.info('Recording has ended.')
                self.g_pool.seek_control.play = False
        elif ts_idx < last_index or ts_idx > last_index + 1:
            # time to seek
            self.seek_to_frame(ts_idx)

        # Only call get_frame() if the next frame is actually needed
        frame = frame or self.get_frame()
        self._recent_frame = frame
        events['frame'] = frame

    def recent_events_own_timing(self, events):
        try:
            frame = self.get_frame()
        except IndexError:
            logger.info('Recording has ended.')
            self.play = False
        else:
            if self.timing:
                self.wait(frame.timestamp)
            self._recent_frame = frame
            events['frame'] = frame

    def get_frame(self):
        try:
            timestamp = self.timestamps[self.target_frame_idx]
        except IndexError:
            raise EndofVideoError('Reached end of timestamps list.')
        except TypeError:
            timestamp = self._recent_wait_ts + 1 / self.fps

        frame = Frame(timestamp, self._img.copy(), self.target_frame_idx)

        frame_txt_font_name = cv2.FONT_HERSHEY_SIMPLEX
        frame_txt_font_scale = 1.
        frame_txt_thickness = 1

        # first line: frame index
        frame_txt = "Fake source frame {}".format(frame.index)
        frame_txt_size = cv2.getTextSize(frame_txt, frame_txt_font_name,
                                         frame_txt_font_scale,
                                         frame_txt_thickness)[0]

        frame_txt_loc = (self.frame_size[0] // 2 - frame_txt_size[0] // 2,
                         self.frame_size[1] // 2 - frame_txt_size[1])

        cv2.putText(frame.img, frame_txt, frame_txt_loc, frame_txt_font_name,
                    frame_txt_font_scale, (255, 255, 255),
                    thickness=frame_txt_thickness, lineType=cv2.LINE_8)

        # second line: resolution @ fps
        frame_txt = "{}x{} @ {} fps".format(*self.frame_size, self.frame_rate)
        frame_txt_size = cv2.getTextSize(frame_txt, frame_txt_font_name,
                                         frame_txt_font_scale,
                                         frame_txt_thickness)[0]

        frame_txt_loc = (self.frame_size[0] // 2 - frame_txt_size[0] // 2,
                         self.frame_size[1] // 2 + frame_txt_size[1])

        cv2.putText(frame.img, frame_txt, frame_txt_loc, frame_txt_font_name,
                    frame_txt_font_scale, (255, 255, 255),
                    thickness=frame_txt_thickness, lineType=cv2.LINE_8)

        self.current_frame_idx = self.target_frame_idx
        self.target_frame_idx += 1

        return frame

    def get_frame_count(self):
        try:
            return len(self.timestamps)
        except TypeError:
            return self.current_frame_idx + 1

    def seek_to_frame(self, frame_idx):
        self.target_frame_idx = max(0, min(frame_idx, self.get_frame_count() - 1))
        self.finished_sleep = 0

    def get_frame_index(self):
        return self.current_frame_idx

    @property
    def name(self):
        return self._name

    @property
    def settings(self):
        return {'frame_size': self.frame_size, 'frame_rate': self.frame_rate}

    @settings.setter
    def settings(self, settings):
        self.frame_size = settings.get('frame_size', self.frame_size)
        self.frame_rate = settings.get('frame_rate', self.frame_rate)

    @property
    def frame_size(self):
        return self._img.shape[1], self._img.shape[0]

    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size
        sizes = [abs(r[0]-new_size[0]) for r in self.frame_sizes]
        best_size_idx = sizes.index(min(sizes))
        size = self.frame_sizes[best_size_idx]
        if size != new_size:
            logger.warning("%s resolution capture mode not available. Selected %s."%(new_size,size))
        self.make_img(size)

    @property
    def frame_rates(self):
        return (30, 60, 90, 120)

    @property
    def frame_sizes(self):
        return ((640, 480), (1280, 720), (1920, 1080))

    @property
    def frame_rate(self):
        return self.fps

    @frame_rate.setter
    def frame_rate(self, new_rate):
        rates = [abs(r-new_rate) for r in self.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning("%sfps capture mode not available at (%s) on 'Fake Source'. Selected %sfps. "%(new_rate, self.frame_size, rate))
        self.fps = rate

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return True

    def get_init_dict(self):
        if self.g_pool.app == 'capture':
            d = super().get_init_dict()
            d['frame_size'] = self.frame_size
            d['frame_rate'] = self.frame_rate
            d['name'] = self.name
            return d
        else:
            raise NotImplementedError()


class Fake_Manager(Base_Manager):
    """Simple manager to explicitly activate a fake source"""

    gui_name = 'Test image'

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def init_ui(self):
        self.add_menu()
        from pyglui import ui
        text = ui.Info_Text('Convenience manager to select a fake source explicitly.')

        def activate():
            # a capture leaving is a must stop for recording.
            self.notify_all({'subject': 'recording.should_stop'})
            settings = {}
            settings['timing'] = 'own'
            settings['frame_rate'] = self.g_pool.capture.frame_rate
            settings['frame_size'] = self.g_pool.capture.frame_size
            settings['name'] = 'Fake Source'
            # if the user set fake capture, we dont want it to auto jump back to the old capture.
            if self.g_pool.process == 'world':
                self.notify_all({'subject':'start_plugin',"name":"Fake_Source",'args':settings})
            else:
                self.notify_all({'subject':'start_eye_capture','target':self.g_pool.process, "name":"Fake_Source",'args':settings})

        activation_button = ui.Button('Activate Fake Capture', activate)
        self.menu.extend([text, activation_button])

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self,events):
        pass
