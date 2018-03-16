'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from time import monotonic, sleep
from plugin import Plugin

import gl_utils
from pyglui import cygl
import numpy as np


import logging
logger = logging.getLogger(__name__)


class InitialisationError(Exception):
    pass


class StreamError(Exception):
    pass


class EndofVideoError(Exception):
    pass


class Base_Source(Plugin):
    """Abstract source class

    All source objects are based on `Base_Source`.

    A source object is independent of its matching manager and should be
    initialisable without it.

    Initialization is required to succeed. In case of failure of the underlying capture
    the follow properties need to be readable:

    - name
    - frame_rate
    - frame_size

    The recent_events function is allowed to not add a frame to the `events` object.

    Attributes:
        g_pool (object): Global container, see `Plugin.g_pool`
    """

    uniqueness = 'by_base_class'
    order = .0
    icon_chr = chr(0xe412)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.capture = self
        self._recent_frame = None
        self._intrinsics = None

    def add_menu(self):
        super().add_menu()
        self.menu_icon.order = 0.2

    def recent_events(self, events):
        """Returns None

        Adds events['frame']=Frame(args)
            Frame: Object containing image and time information of the current
            source frame. See `fake_source.py` for a minimal implementation.
        """
        raise NotImplementedError()

    def gl_display(self):
        if self._recent_frame is not None:
            frame = self._recent_frame
            if frame.yuv_buffer is not None:
                self.g_pool.image_tex.update_from_yuv_buffer(frame.yuv_buffer,frame.width,frame.height)
            else:
                self.g_pool.image_tex.update_from_ndarray(frame.bgr)
            gl_utils.glFlush()
        gl_utils.make_coord_system_norm_based()
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based((self.frame_size[1], self.frame_size[0], 3))

    @property
    def name(self):
        raise NotImplementedError()

    def get_init_dict(self):
        return {}

    @property
    def frame_size(self):
        """Summary
        Returns:
            tuple: 2-element tuple containing width, height
        """
        raise NotImplementedError()

    @frame_size.setter
    def frame_size(self, new_size):
        raise NotImplementedError()

    @property
    def frame_rate(self):
        """
        Returns:
            int/float: Frame rate
        """
        raise NotImplementedError()

    @frame_rate.setter
    def frame_rate(self, new_rate):
        pass

    @property
    def jpeg_support(self):
        """
        Returns:
            bool: Source supports jpeg data
        """
        raise NotImplementedError()

    @property
    def online(self):
        """
        Returns:
            bool: Source is avaible and streaming images.
        """
        return True

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        self._intrinsics = model


class Base_Manager(Plugin):
    """Abstract base class for source managers.

    Managers are plugins that enumerate and load accessible sources from
    different backends, e.g. locally USB-connected cameras.

    Attributes:
        gui_name (str): String used for manager selector labels
    """

    uniqueness = 'by_base_class'
    gui_name = 'Base Manager'
    icon_chr = chr(0xec01)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def add_menu(self):
        super().add_menu()
        from . import manager_classes
        from pyglui import ui

        self.menu_icon.order = 0.1

        def replace_backend_manager(manager_class):
            if self.g_pool.process.startswith('eye'):
                self.g_pool.capture_manager.deinit_ui()
                self.g_pool.capture_manager.cleanup()
                self.g_pool.capture_manager = manager_class(self.g_pool)
                self.g_pool.capture_manager.init_ui()
            else:
                self.notify_all({'subject': 'start_plugin', 'name': manager_class.__name__})

        # We add the capture selection menu
        manager_classes.sort(key=lambda x: x.gui_name)
        self.menu.append(ui.Selector(
                            'capture_manager',
                            setter    = replace_backend_manager,
                            getter    = lambda: self.__class__,
                            selection = manager_classes,
                            labels    = [b.gui_name for b in manager_classes],
                            label     = 'Manager'
                        ))

        # here is where you add all your menu entries.
        self.menu.label = "Backend Manager"


class Playback_Source(Base_Source):

    def __init__(self, g_pool, timing='own', *args, **kwargs):
        '''
        The `timing` argument defines the source's behavior during recent_event calls
            'own': Timing is based on recorded timestamps; uses own wait function;
                    used in Capture as an online source
            'external': Uses Seek_Control's current playback time to figure out
                    most appropriate frame; does not wait on its own
            None: Simply returns next frame as fast as possible; used for detectors
        '''
        super().__init__(g_pool)
        assert timing in ('external', 'own', None), 'invalid timing argument: {}'.format(timing)
        self.timing = timing
        self.finished_sleep = 0.
        self._recent_wait_ts = -1
        self.play = True

    def seek_to_frame(self, frame_idx):
        raise NotImplementedError()

    def get_frame_index(self):
        raise NotImplementedError()

    def get_frame(self):
        raise NotImplementedError()

    def get_frame_index_ts(self):
        idx = self.get_frame_index()
        return idx, self.timestamps[idx]

    def wait(self, timestamp):
        if timestamp == self._recent_wait_ts:
            sleep(1/60)  # 60 fps on pause
        elif self.finished_sleep:
            target_wait_time = timestamp - self._recent_wait_ts
            time_spent = monotonic() - self.finished_sleep
            target_wait_time -= time_spent
            if 1 > target_wait_time > 0:
                sleep(target_wait_time)
        self._recent_wait_ts = timestamp
        self.finished_sleep = monotonic()
