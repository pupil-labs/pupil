'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np
from time import time,sleep

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
    def __init__(self, timestamp,img,index):
        self.timestamp = timestamp
        self.img = img
        self.bgr = img
        self.height,self.width,_ = img.shape
        self._gray = None
        self.index = index
        #indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        return self._gray
    @gray.setter
    def gray(self, value):
        raise Exception('Read only.')

class Fake_Capture(object):
    """docstring for FakeCapture"""
    def __init__(self):
        super(Fake_Capture, self).__init__()
        self.fps = 30
        self.presentation_time = time()
        self.make_img((640,480))
        self.frame_count = 0
        self.controls = []



    def make_img(self,size):
        c_w ,c_h = max(1,size[0]/30),max(1,size[1]/30)
        coarse = np.random.randint(0,200,size=(c_h,c_w,3)).astype(np.uint8)
        # coarse[:,:,1] /=5
        # coarse[:,:,2] *=0
        # coarse[:,:,1] /=30
        # self.img = np.ones((size[1],size[0],3),dtype=np.uint8)
        self.img = cv2.resize(coarse,size,interpolation=cv2.INTER_LANCZOS4)


    def get_frame(self):
        now =  time()
        spent = now - self.presentation_time
        wait = max(0,1./self.fps - spent)
        sleep(wait)
        self.presentation_time = time()
        frame_count = self.frame_count
        self.frame_count +=1
        return Frame(now,self.img.copy(),frame_count)

    def get_frame_robust(self):
        return self.get_frame()

    @property
    def frame_size(self):
        return self.img.shape[1],self.img.shape[0]
    @frame_size.setter
    def frame_size(self,new_size):
        self.make_img(new_size)

    @property
    def frame_rates(self):
        return range(30,121,30)

    @property
    def frame_sizes(self):
        return ((640,480),(1280,720),(1920,1080))

    @property
    def frame_rate(self):
        return self.fps
    @frame_rate.setter
    def frame_rate(self,new_rate):
        self.fps = new_rate

    @property
    def name(self):
        return 'Fake Capture'
