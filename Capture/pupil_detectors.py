'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License. 
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np
from methods import *
import atb
from ctypes import c_int,c_bool,c_float

class Pupil_Detector(object):
    """base class for pupil detector"""
    def __init__(self):
        super(Pupil_Detector, self).__init__()
        var1 = c_int(0)

    def detect(self,img,roi,p_roi,visualize=False):
        # hint: create a view into the img with the bounds of the coarse pupil estimation
        pupil_img = img[roi.lY:roi.uY,roi.lX:roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]

        if visualize:
            # draw into image whatever you like and it will be displayed
            # otherwise you shall not modify img data inplace!
            pass

        candidate_pupil_ellipse = {'center': (None,None),
                        'axes': (None, None),
                        'angle': None,
                        'area': None,
                        'ratio': None,
                        'major': None,
                        'minor': None,
                        'goodness': 0} #some estimation on how sure you are about the detected ellipse and its fit. Smaller is better

        # If you use region of interest p_r and r make sure to return pupil coordinates relative to the full image
        candidate_pupil_ellipse['center'] = roi.add_vector(p_roi.add_vector(candidate_pupil_ellipse['center']))

        return [candidate_pupil_ellipse,] # return list of candidate pupil ellipses, sorted by certainty, if none is found return empty list


    def create_atb_bar(self,pos):
        self.bar = atb.Bar(name = "Pupil_Detector", label="Controls",
            help="pupil detection params", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 200))
        bar.add_var("VAR1",self.var1, step=1.,readonly=False)



class Canny_Detector(Pupil_Detector):
    """a Pupil detector based on Canny_Edges"""
    def __init__(self):
        super(Canny_Detector, self).__init__()
        self.min_contour_size = 80
        self.bin_thresh = c_int(0)
        self.target_ratio=1.0
        self.target_size=c_float(100.)
        self.goodness = c_float(1.)
        self.size_tolerance=10.
        self.blur = c_int(1)
        self.canny_thresh = c_int(200)
        self.canny_ratio= c_int(2)
        self.canny_aperture = c_int(5)

    def detect(self,img,roi,p_roi,visualize=False):

        pupil_img = img[roi.lY:roi.uY,roi.lX:roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]
        pupil_img = grayscale(pupil_img)

        # binary thresholding of pupil dark areas
        hist = cv2.calcHist([pupil_img],[0],None,[256],[0,256]) #(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        bins = np.arange(hist.shape[0])
        spikes = bins[hist[:,0]>40] # every color seen in more than 40 pixels
        if spikes.shape[0] >0:
            lowest_spike = spikes.min()
        else:
            lowest_spike = 200
        offset = 40

        if visualize:
            # display the histogram
            sx,sy = 100,1
            colors = ((0,0,255),(255,0,0),(255,255,0))
            h,w,chan = img.shape
            hist *= 1./hist.max()  # normalize for display

            for i,h in zip(bins,hist[:,0]):
                c = colors[1]
                cv2.line(img,(w,int(i*sy)),(w-int(h*sx),int(i*sy)),c)
            cv2.line(img,(w,int(lowest_spike*sy)),(int(w-.5*sx),int(lowest_spike*sy)),colors[0])
            cv2.line(img,(w,int((lowest_spike+offset)*sy)),(int(w-.5*sx),int((lowest_spike+offset)*sy)),colors[2])

        # create dark and spectral glint masks
        self.bin_thresh.value = lowest_spike
        binary_img = bin_thresholding(pupil_img,image_upper=lowest_spike+offset)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        cv2.dilate(binary_img, kernel,binary_img, iterations=2)
        spec_mask = bin_thresholding(pupil_img, image_upper=250)
        cv2.erode(spec_mask, kernel,spec_mask, iterations=1)


        if self.blur.value >1:
            pupil_img = cv2.medianBlur(pupil_img,self.blur.value)

        edges = cv2.Canny(pupil_img,
                            self.canny_thresh.value,
                            self.canny_thresh.value*self.canny_ratio.value,
                            apertureSize= self.canny_aperture.value)

        # remove edges in areas not dark enough and where the glint is (spectral refelction from IR leds)
        edges = cv2.min(edges, spec_mask)
        edges = cv2.min(edges,binary_img)

        if visualize:
            overlay =  img[roi.lY:roi.uY,roi.lX:roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]
            pupil_img = grayscale(overlay)
            overlay[:,:,1] = cv2.max(pupil_img,edges) #b channel
            overlay[:,:,0] = cv2.max(pupil_img,binary_img) #g channel
            overlay[:,:,2] = cv2.min(pupil_img,spec_mask) #b channel


        # from edges to contours to ellipses
        contours, hierarchy = cv2.findContours(edges,
                                            mode=cv2.RETR_LIST,
                                            method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
        # contours is a list containing array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )

        # the pupil target size is the one closest to the pupil_roi width or heigth (is the same)
        # self.target_size.value = p_roi.uX-p_roi.lX

        good_contours = [c for c in contours if c.shape[0]>self.min_contour_size]
        shape = edges.shape
        ellipses = ((cv2.fitEllipse(c),c) for c in good_contours)
        ellipses = ((e,c) for e,c in ellipses if (0 < e[0][1] < shape[0] and 0< e[0][0] < shape[1])) # center is inside roi
        ellipses = ((e,c) for e,c in ellipses if binary_img[e[0][1],e[0][0]]) # center is on a dark pixel
        ellipses = [(size_deviation(e,self.target_size.value),e,c) for e,c in ellipses if is_round(e,self.target_ratio)] # roundness test
        result = []
        for size_dif,e,c in ellipses:
            pupil_ellipse = {}
            pupil_ellipse['contour'] = c
            a,b = e[1][0]/2.,e[1][1]/2. # majar minor radii of candidate ellipse
            # pupil_ellipse['circumference'] = np.pi*abs(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
            pupil_ellipse['contour_area'] = cv2.contourArea(c)
            pupil_ellipse['ellipse_area'] = np.pi*a*b
            if abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area']) <10:
                pupil_ellipse['goodness'] = 0 #perfect match we'll take this one
            else:
                pupil_ellipse['goodness'] = size_dif
            pupil_ellipse['center'] = roi.add_vector(p_roi.add_vector(e[0])) # compensate for roi offsets
            pupil_ellipse['angle'] = e[-1]
            pupil_ellipse['axes'] = e[1]
            pupil_ellipse['major'] = max(e[1])
            pupil_ellipse['minor'] = min(e[1])
            pupil_ellipse['ratio'] = pupil_ellipse['minor']/pupil_ellipse['major']
            result.append(pupil_ellipse)


        if result:
            result.sort(key=lambda e: e['goodness'])
            self.target_size.value = result[0]['major']

        result = [r for r in result if r['goodness']<self.size_tolerance]

        if result:
            self.goodness.value = result[0]['goodness']

            if result[0]['goodness'] ==0: # perfect match!
                self.target_size.value = result[0]['major']
            else:
                self.target_size.value  = self.target_size.value +  .5 * (result[0]['major']-self.target_size.value)

        else:
            self.goodness.value = 100

        return result


    def create_atb_bar(self,pos):
        self._bar = atb.Bar(name = "Canny_Pupil_Detector", label="Pupil_Detector",
            help="pupil detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 100))
        self._bar.add_var("Pupil_Aparent_Size",self.target_size)
        self._bar.add_var("Pupil_Shade",self.bin_thresh, readonly=True)
        self._bar.add_var("Pupil_Certainty",self.goodness, readonly=True)
        self._bar.add_var("Image_Blur",self.blur, step=2,min=1,max=9)
        self._bar.add_var("Canny_aparture",self.canny_aperture, step=2,min=3,max=7)
        self._bar.add_var("canny_threshold",self.canny_thresh, step=1,min=0)
        self._bar.add_var("Canny_ratio",self.canny_ratio, step=1,min=1)