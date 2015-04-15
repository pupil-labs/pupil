'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os,sys
import cv2
import numpy as np
from time import time,sleep

from pyglui import ui
from ctypes import c_double
import platform
os_name = platform.system()
del platform

#logging
import logging
logger = logging.getLogger(__name__)


class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(FileCaptureError, self).__init__()
        self.arg = arg

class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,img):
        self.timestamp = timestamp
        self.img = img
        self.height,self.width,_ = img.shape
        self._gray = None
        self._yuv = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        return self._gray
    @gray.setter
    def gray(self, value):
        raise Exception('Read only.')

class FakeCapture(object):
    """docstring for FakeCapture"""
    def __init__(self, size=(640,480),fps=30,timestamps=None,timebase=None):
        super(FakeCapture, self).__init__()
        self.size = size
        self.fps = fps
        self.timestamps = timestamps
        self.presentation_time = time()

        self.make_img()

        self.sidebar = None
        self.menu = None

        if timebase == None:
            logger.debug("Capture will run with default system timebase")
            self.timebase = c_double(0)
        elif hasattr(timebase,'value'):
            logger.debug("Capture will run with app wide adjustable timebase")
            self.timebase = timebase
        else:
            logger.error("Invalid timebase variable type. Will use default system timebase")
            self.timebase = c_double(0)

    def make_img(self):
        c_w ,c_h = max(1,self.size[0]/20),max(1,self.size[1]/20)
        coarse = np.random.randint(0,255,size=(c_h,c_w,3)).astype(np.uint8)
        # self.img = np.ones((size[1],size[0],3),dtype=np.uint8)
        self.img = cv2.resize(coarse,self.size,interpolation=cv2.INTER_NEAREST)

    def fastmode(self):
        self.fps = 2000

    def get_frame(self):
        now =  time()
        spent = now - self.presentation_time
        wait = max(0,1./self.fps - spent)
        sleep(wait)
        self.presentation_time = time()
        return Frame(time()-self.timebase.value,self.img.copy())

    @property
    def frame_size(self):
        return self.size

    @property
    def frame_rate(self):
        return self.fps

    def get_now(self):
        return time()


    def init_gui(self,sidebar):

        #create the menu entry
        self.menu = ui.Growing_Menu(label='Camera Settings')
        self.menu.append(ui.Slider('fps',self,min=5,max=500,step=1))
        self.menu.collapsed = True
        self.sidebar = sidebar
        self.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None

    def close(self):
        pass