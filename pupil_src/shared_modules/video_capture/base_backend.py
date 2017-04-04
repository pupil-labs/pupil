'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import Plugin

# imports need to
import gl_utils
from pyglui import cygl
import numpy as np


import logging
logger = logging.getLogger(__name__)


class InitialisationError(Exception):
    def __init__(self, msg=None):
        super().__init__()
        self.message = msg


class StreamError(Exception):
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

    def __init__(self, g_pool):
        assert(not isinstance(g_pool, dict))
        super().__init__(g_pool)
        self.g_pool.capture = self
        self._recent_frame = None

    def cleanup(self):
        self.deinit_gui()

    def init_gui(self):
        """Place to add UI to system-provided menu

        System creates `self.g_pool.capture_source_menu`. UI elements
        should go in there. Only called once and if UI is supported.

        e.g. self.g_pool.capture_source_menu.extend([])
        """
        pass

    def deinit_gui(self):
        """By default, removes all UI elements from system-provided menu

        Only called once and if UI is supported.
        """
        try:
            del self.g_pool.capture_source_menu[:]
        except AttributeError:
            pass

    def recent_events(self, events):
        """Returns None

        Adds events['frame']=Frame(args)
            Frame: Object containing image and time information of the current
            source frame. See `fake_source.py` for a minimal implementation.
        """
        raise NotImplementedError()

    def gl_display(self):
        if self._recent_frame is not None:
            self.g_pool.image_tex.update_from_frame(self._recent_frame)
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


class Base_Manager(Plugin):
    """Abstract base class for source managers.

    Managers are plugins that enumerate and load accessible sources from
    different backends, e.g. locally USB-connected cameras.

    Attributes:
        gui_name (str): String used for manager selector labels
    """

    uniqueness = 'by_base_class'
    gui_name = 'Base Manager'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.capture_manager = self

    def get_init_dict(self):
        return {}

    def init_gui(self):
        """GUI initialisation, see `Plugin.init_gui`

        UI elements should be placed in `self.g_pool.capture_selector_menu`
        """
        pass

    def deinit_gui(self):
        """Removes GUI elements but backend selector"""
        del self.g_pool.capture_selector_menu[1:]

    def cleanup(self):
        self.deinit_gui()
