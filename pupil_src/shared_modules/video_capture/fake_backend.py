'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from .base_backend import Base_Source, Base_Manager

import cv2
import numpy as np
from time import time,sleep
from pyglui import ui

#logging
import logging
logger = logging.getLogger(__name__)


class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,img,index):
        self.timestamp = timestamp
        self._img = img
        self.bgr = img
        self.height,self.width,_ = img.shape
        self._gray = None
        self.index = index
        #indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    @property
    def img(self):
        return self._img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self._img,cv2.COLOR_BGR2GRAY)
        return self._gray
    @gray.setter
    def gray(self, value):
        raise Exception('Read only.')

class Fake_Source(Base_Source):
    """Simple source which shows random, static image.

    It is used as falback in case the original source fails. `preferred_source`
    contains the necessary information to recover to the original source if
    it becomes accessible again.

    Attributes:
        frame_count (int): Sequence counter
        frame_rate (int)
        frame_size (tuple)
    """
    def __init__(self, g_pool, name,frame_size,frame_rate):
        super().__init__(g_pool)
        self.fps = frame_rate
        self._name = name
        self.presentation_time = time()
        self.make_img(tuple(frame_size))
        self.frame_count = 0

    def init_gui(self):
        from pyglui import ui
        text = ui.Info_Text("Fake capture source streaming test images.")
        self.g_pool.capture_source_menu.append(text)

    def cleanup(self):
        self.info_text = None
        self.preferred_source = None
        super().cleanup()

    def make_img(self,size):
        c_w ,c_h = max(1,size[0]/30),max(1,size[1]/30)
        coarse = np.random.randint(0,200,size=(int(c_h),int(c_w),3)).astype(np.uint8)
        # coarse[:,:,1] /=5
        # coarse[:,:,2] *=0
        # coarse[:,:,1] /=30
        # self._img = np.ones((size[1],size[0],3),dtype=np.uint8)
        self._img = cv2.resize(coarse,size,interpolation=cv2.INTER_LANCZOS4)

    def recent_events(self,events):
        now = time()
        spent = now - self.presentation_time
        wait = max(0,1./self.fps - spent)
        sleep(wait)
        self.presentation_time = time()
        self.frame_count +=1
        timestamp = self.g_pool.get_timestamp()
        frame = Frame(timestamp,self._img.copy(),self.frame_count)
        cv2.putText(frame.img, "Fake Source Frame %s"%self.frame_count,(20,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,100,100))
        events['frame'] = frame
        self._recent_frame = frame

    @property
    def name(self):
        return self._name

    @property
    def settings(self):
        return self.preferred_source

    @settings.setter
    def settings(self,settings):
        self.frame_size = settings.get('frame_size', self.frame_size)
        self.frame_rate = settings.get('frame_rate', self.frame_rate )

    @property
    def frame_size(self):
        return self._img.shape[1],self._img.shape[0]
    @frame_size.setter
    def frame_size(self,new_size):
        #closest match for size
        sizes = [ abs(r[0]-new_size[0]) for r in self.frame_sizes ]
        best_size_idx = sizes.index(min(sizes))
        size = self.frame_sizes[best_size_idx]
        if size != new_size:
            logger.warning("%s resolution capture mode not available. Selected %s."%(new_size,size))
        self.make_img(size)

    @property
    def frame_rates(self):
        return (30,60,90,120)

    @property
    def frame_sizes(self):
        return ((640,480),(1280,720),(1920,1080))

    @property
    def frame_rate(self):
        return self.fps
    @frame_rate.setter
    def frame_rate(self,new_rate):
        rates = [ abs(r-new_rate) for r in self.frame_rates ]
        best_rate_idx = rates.index(min(rates))
        rate = self.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning("%sfps capture mode not available at (%s) on 'Fake Source'. Selected %sfps. "%(new_rate,self.frame_size,rate))
        self.fps = rate

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return True

    def get_init_dict(self):
        d = super().get_init_dict()
        d['frame_size'] = self.frame_size
        d['frame_rate'] = self.frame_rate
        d['name'] = self.name
        return d


class Fake_Manager(Base_Manager):
    """Simple manager to explicitly activate a fake source"""

    gui_name = 'Test image'

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def init_gui(self):
        from pyglui import ui
        text = ui.Info_Text('Convenience manager to select a fake source explicitly.')

        def activate():
            #a capture leaving is a must stop for recording.
            self.notify_all( {'subject':'recording.should_stop'} )
            settings = {}
            settings['frame_rate'] = self.g_pool.capture.frame_rate
            settings['frame_size'] = self.g_pool.capture.frame_size
            settings['name'] = self.g_pool.capture.name
            #if the user set fake capture, we dont want it to auto jump back to the old capture.
            if self.g_pool.process == 'world':
                self.notify_all({'subject':'start_plugin',"name":"Fake_Source",'args':settings})
            else:
                self.notify_all({'subject':'start_eye_capture','target':self.g_pool.process, "name":"Fake_Source",'args':settings})

        activation_button = ui.Button('Activate Fake Capture', activate)
        self.g_pool.capture_selector_menu.extend([text, activation_button])


    def recent_events(self,events):
        pass