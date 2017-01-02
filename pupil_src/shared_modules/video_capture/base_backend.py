'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2017  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the files COPYING and COPYING.LESSER, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import logging
logger = logging.getLogger(__name__)

class InitialisationError(Exception):
    def __init__(self,msg=None):
        super(InitialisationError,self).__init__()
        self.message = msg

class StreamError(Exception):
    pass

class Base_Source(object):
    """Abstract source class

    All source objects are based on `Base_Source`. It defines a basic interface
    which is a variation of `Plugin`.

    A source object is independent of its matching manager and should be
    initialisable without it. If something fails the system will fallback
    to `Fake_Source` which will wrap the settings of the previous source.
    This feature can be used for re-initialisation of the previous source in
    case it is accessible again. See `Base_Manager` for more information on
    source recovery.

    Attributes:
        g_pool (object): Global container, see `Plugin.g_pool`
    """

    def __init__(self, g_pool):
        assert(not isinstance(g_pool, dict))
        super(Base_Source, self).__init__()
        self.g_pool = g_pool

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
        del self.g_pool.capture_source_menu[:]

    def cleanup(self):
        """Called on source termination."""
        pass

    def get_frame(self):
        """Returns the current frame object

        Returns:
            Frame: Object containing image and time information of the current
            source frame. See `fake_source.py` for a minimal implementation.
        """
        raise NotImplementedError()

    def notify_all(self,notification):
        """Same as `Plugin.notify_all`"""
        self.g_pool.ipc_pub.notify(notification)

    def on_notify(self,notification):
        """Same as `Plugin.on_notify`"""
        pass

    def gl_display(self):
        """Same as `Plugin.gl_display`"""
        pass

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def settings(self):
        """Dict containing recovery information.

        Subclasses should extend the minimal set of recovery information below.

        Returns:
            dict: Recovery information
        """
        return {
            'source_class_name': self.class_name(),
            'name': self.name
        }

    @settings.setter
    def settings(self,settings):
        """Allows to use selective information from settings

        Args: settings (dict)
        """
        pass

    @property
    def frame_size(self):
        """Summary

        Returns:
            tuple: 2-element tuple containing width, height
        """
        raise NotImplementedError()
    @frame_size.setter
    def frame_size(self,new_size):
        raise NotImplementedError()

    @property
    def frame_rate(self):
        """
        Returns:
            int/float: Frame rate
        """
        raise NotImplementedError()
    @frame_rate.setter
    def frame_rate(self,new_rate):
        pass

    @property
    def jpeg_support(self):
        """
        Returns:
            bool: Source supports jpeg data
        """
        raise NotImplementedError()


    @classmethod
    def class_name(self):
        return self.__name__

class Base_Manager(Plugin):
    """Abstract base class for source managers.

    Managers are plugins that enumerate and load accessible sources from
    different backends, e.g. locally USB-connected cameras. They should notify
    other plugins about new and disconnected sources using the
    `capture_manager.source_found` and `capture_manager.source_lost`
    notifications.

    Managers are able to activate sources. The default behaviour is to only
    activate a new source if it is accessible.

    In case a fake source is active, it is possible to try to recover to the
    original source whose settings are stored in `Fake_Source.preferred_source`

    Attributes:
        gui_name (str): String used for manager selector labels
    """

    uniqueness = 'by_base_class'
    gui_name = '???'

    def __init__(self, g_pool):
        super(Base_Manager, self).__init__(g_pool)
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