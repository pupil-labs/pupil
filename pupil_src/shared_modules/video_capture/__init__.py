'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
video_capture is a module that extends opencv's camera_capture for mac and windows
on Linux it repleaces it completelty.
it adds some fuctionalty like:
    - access to all uvc controls
    - assosication by name patterns instead of id's (0,1,2..)
it requires:
    - opencv 2.3+
    - on Linux: v4l2-ctl (via apt-get install v4l2-util)
    - on MacOS: uvcc (binary is distributed with this module)
"""
import os,sys,traceback as tb
import cv2
import numpy as np
from os.path import isfile
from time import time

from pyglui import ui

import platform
os_name = platform.system()
del platform

#logging
import logging
logger = logging.getLogger(__name__)

###OS specific imports and defs
if os_name in ("Linux","Darwin"):
    from uvc import is_accessible as uvc_is_accessible, device_list as uvc_device_list
    from uvc_capture import Camera_Capture,CameraCaptureError
elif os_name == "Windows":
    from win_video import Camera_Capture,device_list as uvc_device_list,CameraCaptureError
    def uvc_is_accessible(uid):
        return True
else:
    raise NotImplementedError()

from av_file_capture import File_Capture, FileCaptureError, EndofVideoFileError,FileSeekError

from ndsi_capture import Network_Device_Manager
from fake_capture import Fake_Capture

class Capture_Manager(object):
    """docstring for Capture_Manager"""
    def __init__(self,g_pool,previous_settings,fallback_settings):
        super(Capture_Manager, self).__init__()
        self.g_pool = g_pool
        self.ndm = Network_Device_Manager()
        self.cap_uid = None
        self.cap_name = None
        self.cap_type = None

        if not self.init_fields_from_settings(previous_settings) and \
            not self.init_fields_from_settings(fallback_settings):
            self.cap_uid = None
        self.init_capture_from_fields()

        other_settings = fallback_settings.copy()
        if previous_settings:
            other_settings.update(previous_settings)
        self.settings = other_settings

        self.sidebar = None
        self.menu = None
        self.source_selector = None
        self.control_menu = None

    def init_fields_from_settings(self, settings):
        if not settings: return False

        self.cap_type = settings.get('cap_type')
        if not self.cap_type: return False

        if self.cap_type == 'file':
            self.path = kwargs.get('path')
            self.timestamps = kwargs.get('timestamps')
        elif self.cap_type == 'uvc':
            self.cap_name = settings.get('cap_name')
            if self.cap_name:
                names = [self.cap_name]
            else:
                names = settings.get('names',[])
            dev = self.accessible_dev_from_list(names)
            if not dev: return False
            self.cap_name = dev['name']
            self.cap_uid  = dev['uid']
        elif self.cap_type == 'ndsi':
            pass
        return self.init_capture_from_fields()

    def init_capture_from_fields(self):
        try:
            if not self.cap_uid:
                self.capture = Fake_Capture()
                return True
            if self.cap_type == 'file':
                self.capture = File_Capture(self.path,timestamps=self.timestamps)
                return bool(self.capture)
            elif self.cap_type == 'uvc':
                try:
                    self.capture = Camera_Capture(self.g_pool,self.cap_uid)
                except RuntimeError as e:
                    #logger.warning(str(e))
                    pass
                return bool(self.capture)
            elif self.cap_type == 'ndsi':
                return False

        except Exception as e:
            tb.print_exc()
            logger.error('Error initiating capture "%s" of type "%s": %s'%(self.cap_name, self.cap_type, e))
            return False

    def init_cap_by_uid(self,uid):
        new_capture = None
        try:
            if not uid:
                new_capture = Fake_Capture()
            elif self.cap_type == 'file':
                new_capture = File_Capture(uid,timestamps=self.timestamps)
            elif self.cap_type == 'uvc':
                if uvc_is_accessible(uid):
                    new_capture = Camera_Capture(self.g_pool,uid)
                else:
                    logger.error('Selected camera is in use or blocked.')
            elif self.cap_type == 'ndsi':
                logger.warning('Network device "%s" not supported'%uid)
        except:
            new_capture = Fake_Capture()
        if new_capture:
            old_settings = self.capture.settings
            self.capture.close()
            self.capture = new_capture
            self.cap_uid = uid
            self.capture.settings = old_settings
            self.capture.init_gui()
            self.on_frame_size_change(self.capture.frame_size)
        if self.menu:
            self.gui_refresh_sources()

    def accessible_dev_from_list(self, names):
        for name in names:
            for dev in uvc_device_list():
                if name in dev['name']:
                    if uvc_is_accessible(dev['uid']):
                        return dev
                    else: logger.warning("Camera '%s' matches the pattern but is already in use"%name)
        logger.error('No accessible device found that matched %s'%names)

    @property
    def frame_size(self):
        return self.capture.frame_size
    @frame_size.setter
    def frame_size(self,new_size):
        self.capture.frame_size = new_size
        self.on_frame_size_change(size)

    @property
    def frame_rates(self):
        return self.capture.frame_rates

    @property
    def frame_sizes(self):
        return self.capture.frame_sizes

    @property
    def settings(self):
        settings = {}
        if self.capture.__class__ != Fake_Capture:
            settings.update(self.capture.settings)
            settings['cap_type'] = self.cap_type
            if self.cap_type == 'file':
                settings['path'] = self.path
            else:
                settings['cap_name'] = self.cap_name
        return settings
    @settings.setter
    def settings(self,settings):
        self.capture.settings = settings

    @property
    def frame_rate(self):
        return self.capture.frame_rate
    @frame_rate.setter
    def frame_rate(self,new_rate):
        self.capture.frame_rate = new_rate

    def get_frame(self):
        frame = None
        try:
            frame = self.capture.get_frame()
        except CameraCaptureError:
            # Try and re-initialize
            self.capture.close()
            self.init_cap_by_uid(self.capture.cap_uid)
        finally:
            return frame

    def set_capture_control_menu(self):
        if self.control_menu:
            self.menu.remove(self.control_menu)
        self.control_menu = self.capture.control_menu
        self.menu.insert(2, self.control_menu)

    def gui_refresh_sources(self):
        def init_source_selector():
            if self.cap_type == 'file':
                return Text_Input('path',self,label='Capture file',setter=self.init_cap_by_uid)
            else:
                if self.cap_type == 'uvc':
                    label = 'UVC device'
                    cameras = uvc_device_list()
                elif self.cap_type == 'ndsi':
                    label = 'Network device'
                    cameras = self.ndm.device_list()
                camera_names = ['Fake Capture']+[c['name'] for c in cameras]
                camera_ids = [None]+[c['uid'] for c in cameras]
                return ui.Selector('cap_uid',self,selection=camera_ids,labels=camera_names,label=label, setter=self.init_cap_by_uid)

        selector = init_source_selector()
        if self.source_selector:
            self.menu.remove(self.source_selector)
        self.source_selector = selector
        self.menu.insert(1, selector)

    def init_gui(self,sidebar):

        def gui_set_source_type(src_type):
            if self.cap_type == src_type:
                return
            self.uid = None
            self.cap_type = src_type
            self.gui_refresh_sources()
            self.set_capture_control_menu()

        self.capture.init_gui()
        self.menu = ui.Growing_Menu(label='Camera Settings')
        types = ['file','uvc','ndsi']
        type_labels = ['File', 'UVC', 'Network']
        self.menu.append(ui.Selector('cap_type',self,selection=types,labels=type_labels,label='Source Type', setter=gui_set_source_type) )
        self.gui_refresh_sources()
        self.set_capture_control_menu()

        self.sidebar = sidebar
        #add below geneal settings
        self.sidebar.insert(1,self.menu)

    def on_frame_size_change(self, new_size):
        pass

    def deinit_gui(self):
        if self.menu:
            if self.source_selector:
                self.menu.remove(self.source_selector)
            if self.control_menu:
                self.menu.remove(self.control_menu)
                self.control_menu = None
            self.sidebar.remove(self.menu)
            self.menu = None
        self.sidebar = None


    def close(self):
        self.deinit_gui()
        self.capture.close()