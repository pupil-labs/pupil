
'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# make shared modules available across pupil_src
if __name__ == '__main__':
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath

import cv2
from time import sleep
import numpy as np
from methods import *
import atb
from ctypes import c_int,c_bool,c_float
import logging
logger = logging.getLogger(__name__)
from c_methods import eye_filter
from template import Pupil_Detector

class MSER_Detector(Pupil_Detector):
    """docstring for MSER_Detector"""
    def __init__(self):
        super(MSER_Detector, self).__init__()

    def detect(self,frame,u_roi,visualize=False):
        #get the user_roi
        img = frame.img
        # r_img = img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX]
        debug= True
        PARAMS = {'_delta':10, '_min_area': 2000, '_max_area': 10000, '_max_variation': .25, '_min_diversity': .2, '_max_evolution': 200, '_area_threshold': 1.01, '_min_margin': .003, '_edge_blur_size': 7}
        pupil_intensity= 150
        pupil_ratio= 2
        mser = cv2.MSER(**PARAMS)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions = mser.detect(gray, None)
        hulls = []
        # Select most circular hull
        for region in regions:
            h = cv2.convexHull(region.reshape(-1, 1, 2)).reshape((-1, 2))
            cv2.drawContours(frame.img,[h],-1,(255,0,0))
            hc = h - np.mean(h, 0)
            _, s, _ = np.linalg.svd(hc)
            r = s[0] / s[1]
            if r > pupil_ratio:
                logger.debug('Skipping ratio %f > %f' % (r, pupil_ratio))
                continue
            mval = np.median(gray.flat[np.dot(region, np.array([1, img.shape[1]]))])
            if mval > pupil_intensity:
                logger.debug('Skipping intensity %f > %f' % (mval,pupil_intensity))
                continue
            logger.debug('Kept: Area[%f] Intensity[%f] Ratio[%f]' % (region.shape[0], mval, r))
            hulls.append((r, region, h))
        if hulls:
            hulls.sort()
            gaze = np.round(np.mean(hulls[0][2].reshape((-1, 2)), 0)).astype(np.int).tolist()
            logger.debug('Gaze[%d,%d]' % (gaze[0], gaze[1]))
            norm_pupil = normalize((gaze[0], gaze[1]), (img.shape[1], img.shape[0]),flip_y=True )
            return {'norm_pupil':norm_pupil,'timestamp':frame.timestamp,'center':(gaze[0], gaze[1])}
        else:
            return {'norm_pupil':None,'timestamp':frame.timestamp}



    def create_atb_bar(self,pos):
        self.bar = atb.Bar(name = "MSER_Detector", label="MSER PUPIL Detector Controls",
            help="pupil detection params", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 200))
        # self.bar.add_var("VAR1",self.var1, step=1.,readonly=False)

