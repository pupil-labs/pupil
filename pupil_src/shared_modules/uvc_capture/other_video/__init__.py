'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from cv2 import VideoCapture
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
    def __init__(self, timestamp,img,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.img = img
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt


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
        self.capture = VideoCapture(src_id)
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

    def set_size(self,size):
        width,height = size
        self.capture.set(3, width)
        self.capture.set(4, height)

    def get_size(self):
        return self.capture.get(3), self.capture.get(4)

    def set_fps(self,fps):
        self.capture.set(5,fps)

    def get_fps(self):
        return self.capture.get(5)

    def get_now(self):
        return time()

    def create_atb_bar(self,pos):
        size = 0,0
        return size

    def kill_atb_bar(self):
        pass

    def close(self):
        pass
