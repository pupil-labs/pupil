'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

class Base_Source(object):
    """docstring for Base_Source"""
    def __init__(self, g_pool, on_frame_size_change=None):
        super(Base_Source, self).__init__()
        self.g_pool = g_pool
        self.parent_menu = None

