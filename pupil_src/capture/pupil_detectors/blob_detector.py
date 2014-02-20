
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

class Blob_Detector(Pupil_Detector):
    """a Pupil detector based on Canny_Edges"""
    def __init__(self):
        super(Blob_Detector, self).__init__()
        self.intensity_range = c_int(18)
        self.canny_thresh = c_int(200)
        self.canny_ratio= c_int(2)
        self.canny_aperture = c_int(5)


    def detect(self,frame,u_roi,visualize=False):
        #get the user_roi
        img = frame.img
        r_img = img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX]
        gray_img = grayscale(r_img)
        # coarse pupil detection
        integral = cv2.integral(gray_img)
        integral =  np.array(integral,dtype=c_float)
        x,y,w,response = eye_filter(integral,100,400)
        p_roi = Roi(gray_img.shape)
        if w>0:
            p_roi.set((y,x,y+w,x+w))
        else:
            p_roi.set((0,0,-1,-1))
        coarse_pupil_center = x+w/2.,y+w/2.
        coarse_pupil_width = w/2.
        padding = coarse_pupil_width/4.
        pupil_img = gray_img[p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]
        # binary thresholding of pupil dark areas
        hist = cv2.calcHist([pupil_img],[0],None,[256],[0,256]) #(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        bins = np.arange(hist.shape[0])
        spikes = bins[hist[:,0]>40] # every intensity seen in more than 40 pixels
        if spikes.shape[0] >0:
            lowest_spike = spikes.min()
            highest_spike = spikes.max()
        else:
            lowest_spike = 200
            highest_spike = 255


        offset = self.intensity_range.value
        spectral_offset = 5
        if visualize:
            # display the histogram
            sx,sy = 100,1
            colors = ((0,0,255),(255,0,0),(255,255,0),(255,255,255))
            h,w,chan = img.shape
            hist *= 1./hist.max()  # normalize for display

            for i,h in zip(bins,hist[:,0]):
                c = colors[1]
                cv2.line(img,(w,int(i*sy)),(w-int(h*sx),int(i*sy)),c)
            cv2.line(img,(w,int(lowest_spike*sy)),(int(w-.5*sx),int(lowest_spike*sy)),colors[0])
            cv2.line(img,(w,int((lowest_spike+offset)*sy)),(int(w-.5*sx),int((lowest_spike+offset)*sy)),colors[2])
            cv2.line(img,(w,int((highest_spike)*sy)),(int(w-.5*sx),int((highest_spike)*sy)),colors[0])
            cv2.line(img,(w,int((highest_spike- spectral_offset )*sy)),(int(w-.5*sx),int((highest_spike - spectral_offset)*sy)),colors[3])


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

        #open operation to remove eye lashes
        pupil_img = cv2.morphologyEx(pupil_img, cv2.MORPH_OPEN, kernel)


        # PARAMS = {}
        # blob_detector = cv2.SimpleBlobDetector(**PARAMS)
        # kps =  blob_detector.detect(pupil_img)

        # blur = cv2.GaussianBlur(pupil_img,(5,5),0)
        blur = pupil_img
        # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret3,th3 = cv2.threshold(blur,lowest_spike+offset,255,cv2.THRESH_BINARY)
        # ret3,th3 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th3 = cv2.Laplacian(th3,cv2.CV_64F)

        edges = cv2.Canny(pupil_img,
                            self.canny_thresh.value,
                            self.canny_thresh.value*self.canny_ratio.value,
                            apertureSize= self.canny_aperture.value)

        r_img[p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX,1] = th3
        r_img[p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX,2] = edges
        # for kp in kps:
        #     print kp.pt
        #     cv2.circle(r_img[p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX],tuple(map(int,kp.pt,)),10,(255,255,255))

        no_result = {}
        no_result['timestamp'] = frame.timestamp
        no_result['norm_pupil'] = None
        return no_result



    def create_atb_bar(self,pos):
        self._bar = atb.Bar(name = "Canny_Pupil_Detector", label="Pupil_Detector",
            help="pupil detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 100))
        # self._bar.add_var("pupil_intensity_range",self.intensity_range)

