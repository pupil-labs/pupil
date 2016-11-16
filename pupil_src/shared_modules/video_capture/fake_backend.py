'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from .base_backend import Base_Source, Base_Manager

import cv2
import numpy as np
from time import time,sleep
from pyglui import ui

#logging
import logging
logger = logging.getLogger(__name__)

class CameraSourceError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(FileSourceError, self).__init__()
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

class Fake_Source(Base_Source):
    """Simple source which shows random, static image.

    It is used as falback in case the original source fails. `preferred_source`
    contains the necessary information to recover to the original source if
    it becomes accessible again.

    Attributes:
        fps (int)
        frame_count (int): Sequence counter
        frame_rate (int)
        frame_size (tuple)
        img (uint8 numpy array): bgr image
        info_text (str): String shown to the user
        preferred_source (dict): Settings about source which this fake capture is a fallback for.
        presentation_time (float)
        settings (dict)
    """
    def __init__(self, g_pool, **settings):
        super(Fake_Source, self).__init__(g_pool)
        self.fps = 30
        self.presentation_time = time()
        self.make_img((640,480))
        self.frame_count = 0
        self.info_text = settings.get('info_text', 'Select video source in the capture menu.')
        self.preferred_source = settings
        self.settings = settings

    def init_gui(self):
        from pyglui import ui
        text = ui.Info_Text(self.info_text)
        self.g_pool.capture_source_menu.append(text)

        from pyglui.pyfontstash import fontstash
        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',ui.get_opensans_font_path())
        self.glfont.set_size(64)
        self.glfont.set_color_float((1.,1.,1.,1.0))
        self.glfont.set_align_string(v_align='center',h_align='middle')




    def cleanup(self):
        self.info_text = None
        self.img = None
        self.preferred_source = None

    def gl_display(self):
        if hasattr(self,'glfont'):
            from glfw import glfwGetFramebufferSize,glfwGetCurrentContext
            self.window_size = glfwGetFramebufferSize(glfwGetCurrentContext())
            self.glfont.set_color_float((0.,0.,0.,1.))
            self.glfont.set_size(int(self.window_size[0]/30.))
            self.glfont.set_blur(5.)
            self.glfont.draw_limited_text(self.window_size[0]/2.,self.window_size[1]/2.,self.info_text,self.window_size[0]*0.8)
            self.glfont.set_blur(0.96)
            self.glfont.set_color_float((1.,1.,1.,1.0))
            self.glfont.draw_limited_text(self.window_size[0]/2.,self.window_size[1]/2.,self.info_text,self.window_size[0]*0.8)


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

        timestamp = self.g_pool.get_timestamp()
        return Frame(timestamp,self.img.copy(),frame_count)

    @property
    def name(self):
        return self.preferred_source.get('name', 'Fake Source')

    @property
    def settings(self):
        return self.preferred_source

    @settings.setter
    def settings(self,settings):
        self.frame_size = settings.get('frame_size', self.frame_size)
        self.frame_rate = settings.get('frame_rate', self.frame_rate )

    @property
    def frame_size(self):
        return self.img.shape[1],self.img.shape[0]
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
        return range(30,60,120)

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

class Fake_Manager(Base_Manager):
    """Simple manager to explicitly activate a fake source"""

    gui_name = 'Test image'

    def __init__(self, g_pool):
        super(Fake_Manager, self).__init__(g_pool)

    def init_gui(self):
        from pyglui import ui
        text = ui.Info_Text('Convenience manager to select a fake source explicitly.')
        def activate():
            settings = self.g_pool.capture.settings
            #if the user set fake capture, we dont want it to auto jump back to the old capture.
            settings['source_class_name'] = Fake_Source.class_name()
            self.activate_source(settings)
        activation_button = ui.Button('Activate Test Image', activate)
        self.g_pool.capture_selector_menu.extend([text, activation_button])

    def activate_source(self, settings={}):
        self.g_pool.capture.deinit_gui()
        self.g_pool.capture.cleanup()
        self.g_pool.capture = None
        self.g_pool.capture = Fake_Source(self.g_pool,**settings)
        self.g_pool.capture.init_gui()
