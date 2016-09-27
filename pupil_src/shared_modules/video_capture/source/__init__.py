'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

class InitialisationError(Exception):
    pass

class StreamError(Exception):
    pass

class Base_Source(object):
    """Abstract source class

    All source objects are based on `Base_Source`. It defines a basic interface
    which is a variation of `Plugin`.

    A source object is independent of its matching manager and should be
    initialisable without it. If something fails the system will fallback
    to `Fake_Source` which will wrap the settings of the previous source.
    This feature is used for re-initialisation of the previous source in
    case it is accessible again. See `../manager/__init__.py` for more
    information on source recovery.

    Attributes:
        g_pool (object): Global container, see `Plugin.g_pool`
    """

    def __init__(self, g_pool):
        super(Base_Source, self).__init__()
        self.g_pool = g_pool

    def init_gui(self):
        """Place to add UI to system-provided menu

        System creates `self.g_pool.capture_source_menu`. UI elements
        should go in there. Only called once and if UI is supported.
        """
        self.g_pool.capture_source_menu.extend([])

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
    def settings(self):
        """Dict containing recovery information.

        Subclasses should extend the minimal set of recovery information below.

        Returns:
            dict: Recovery information
        """
        return {
            'source_class_name': self.class_name(),
            'name': self.class_name().replace('_',' ')
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
        # Subclasses need to call this:
        self.g_pool.on_frame_size_change(new_size)
        # eye.py sets a custom `on_frame_size_change` callback
        # which recalculates the size of the ROI. If this does not
        # happen, the eye process will crash.

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

    @staticmethod
    def error_class():
        """
        Returns:
            type: Error class which should be caught to initialise fallback
        """
        return StreamError

    @classmethod
    def class_name(self):
        return self.__name__

from fake_source import Fake_Source
from uvc_source  import UVC_Source
from ndsi_source import NDSI_Source
from file_source import File_Source