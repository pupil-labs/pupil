'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
from plugin import Plugin
import numpy as np
import atb
from ctypes import c_int
from methods import denormalize
import logging
logger = logging.getLogger(__name__)

class Vis_Light_Points(Plugin):
    """docstring
    show gaze dots at light dots on numpy.

    """
    #let the plugin work after most other plugins.

    def __init__(self, g_pool=None):
        super(Vis_Light_Points, self).__init__()
        self.order = .8

    def update(self,frame,recent_pupil_positions,events):

        #since we edit the img inplace we should not do it in pause mode...
        img = frame.img
        img_shape = img.shape[:-1][::-1]#width,height
        norm_gaze = [ng['norm_gaze'] for ng in recent_pupil_positions if ng['norm_gaze'] is not None]
        screen_gaze = [denormalize(ng,img_shape,flip_y=True) for ng in norm_gaze]


        overlay = np.ones(img.shape[:-1],dtype=img.dtype)

        # draw recent gaze postions as black dots on an overlay image.
        for gaze_point in screen_gaze:
            try:
                overlay[int(gaze_point[1]),int(gaze_point[0])] = 0
            except:
                pass

        out = cv2.distanceTransform(overlay,cv2.cv.CV_DIST_L2, 5)

        # fix for opencv binding incositency
        if type(out)==tuple:
            out = out[0]

        overlay =  1/(out/20+1)

        img *= cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)


    def init_gui(self,pos=None):
        pos = 10,470
        import atb
        from time import time

        atb_label = "Light Points"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="circle", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.1, size=(300, 20))

        self._bar.add_button('remove',self.unset_alive)

    def unset_alive(self):
        self.alive = False


    def get_init_dict(self):
        return {}

    def clone(self):
        return Vis_Light_Points(**self.get_init_dict())





    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()
