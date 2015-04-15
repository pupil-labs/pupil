'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from time import time

#logging
import logging
logger = logging.getLogger(__name__)

from ctypes import c_double

class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(CameraCaptureError, self).__init__()
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




class Camera_List(list):
        """just an empty list for now
        Cam list enumerates all attached devices
        """
        def __init__(self):
            pass

class Camera_Capture():
    """
    VideoCapture without uvc control using cv2.VideoCapture
    """
    def __init__(self,src_id,size=(640,480),fps=None,timebase=None):
        self.controls = None
        self.cvId = src_id
        self.name = "VideoCapture"
        self.controls = None
        ###add cv videocapture capabilities
        self.capture = cv2.VideoCapture(src_id)
        self.set_size(size)

        if timebase == None:
            logger.debug("Capture will run with default system timebase")
            self.timebase = c_double(0)
        elif isinstance(timebase,c_double):
            logger.debug("Capture will run with app wide adjustable timebase")
            self.timebase = timebase
        else:
            logger.error("Invalid timebase variable type. Will use default system timebase")
            self.timebase = c_double(0)


    def get_frame(self):
        s, img = self.capture.read()
        timestamp = time()
        return Frame(timestamp,img)


    @property
    def frame_size(self):
        return self.capture.get(3), self.capture.get(4)
    @frame_size.setter
    def frame_size(self, value):
        width,height = size
        self.capture.set(3, width)
        self.capture.set(4, height)

    @property
    def frame_rate(self):
        #return rate as denominator only
        return self.capture.get(5)
    @frame_rate.setter
    def frame_rate(self, rate):
        self.capture.set(5,fps)

    def get_now(self):
        return time()

    def create_atb_bar(self,pos):
        size = 0,0
        return size

    def kill_atb_bar(self):
        pass

    def close(self):
        pass
