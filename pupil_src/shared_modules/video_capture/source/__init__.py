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
    """docstring for Base_Source"""

    def __init__(self, g_pool):
        super(Base_Source, self).__init__()
        self.g_pool = g_pool

    def init_gui(self):
        self.g_pool.capture_source_menu.extend([])

    def deinit_gui(self):
        del self.g_pool.capture_source_menu[:]

    def cleanup(self):
        pass

    def get_frame(self):
        raise NotImplementedError()

    def notify_all(self,notification):
        self.g_pool.ipc_pub.notify(notification)

    def on_notify(self,notification):
        pass

    def gl_display(self):
        pass

    def on_frame_size_change(self, new_size):
        pass

    @property
    def settings(self):
        return {
            'source_class_name': self.class_name(),
            'name': self.class_name().replace('_',' ')
        }
    @settings.setter
    def settings(self,settings):
        pass

    @property
    def frame_size(self):
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
        raise NotImplementedError()
    @frame_rate.setter
    def frame_rate(self,new_rate):
        pass

    @property
    def jpeg_support(self):
        raise NotImplementedError()

    @staticmethod
    def error_class():
        return StreamError

    @classmethod
    def class_name(self):
        return self.__name__

from fake_source import Fake_Source
from uvc_source  import UVC_Source
from ndsi_source import NDSI_Source
from file_source import File_Source