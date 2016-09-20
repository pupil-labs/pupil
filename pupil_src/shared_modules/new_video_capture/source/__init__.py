'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


class Base_Source(object):
    """docstring for Base_Source"""
    def __init__(self, g_pool, on_frame_size_change=None):
        super(Base_Source, self).__init__()
        self.g_pool = g_pool
        self.parent_menu = None

from fake_source import Fake_Source
from uvc_source  import UVC_Source
from ndsi_source import NDSI_Source