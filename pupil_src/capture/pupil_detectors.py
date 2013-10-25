'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

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
import numpy as np
from methods import *
import atb
from ctypes import c_int,c_bool,c_float
import logging
logger = logging.getLogger(__name__)
from c_methods import eye_filter
import random

class Pupil_Detector(object):
    """base class for pupil detector"""
    def __init__(self):
        super(Pupil_Detector, self).__init__()
        var1 = c_int(0)

    def detect(self,frame,u_roi,p_roi,visualize=False):
        img = frame.img
        # hint: create a view into the img with the bounds of the coarse pupil estimation
        pupil_img = img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]

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

        # If you use region of interest p_roi and u_roi make sure to return pupil coordinates relative to the full image
        candidate_pupil_ellipse['center'] = u_roi.add_vector(p_roi.add_vector(candidate_pupil_ellipse['center']))
        candidate_pupil_ellipse['timestamp'] = frame.timestamp
        result = candidate_pupil_ellipse #we found something
        if result:
            return candidate_pupil_ellipse # all this will be sent to the world process, you can add whateever you need to this.

        else:
            self.goodness.value = 100
            no_result = {}
            no_result['timestamp'] = frame.timestamp
            no_result['norm_pupil'] = None
            return no_result


    def create_atb_bar(self,pos):
        self.bar = atb.Bar(name = "Pupil_Detector", label="Pupil Detector Controls",
            help="pupil detection params", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 200))
        self.bar.add_var("VAR1",self.var1, step=1.,readonly=False)



class MSER_Detector(Pupil_Detector):
    """docstring for MSER_Detector"""
    def __init__(self):
        super(MSER_Detector, self).__init__()

    def detect(self,frame,u_roi,p_roi,visualize=False):
        img = frame.img
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
        self.intensity_range = c_int(17)

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

        # create dark and spectral glint masks
        self.bin_thresh.value = lowest_spike
        binary_img = bin_thresholding(pupil_img,image_upper=lowest_spike + offset)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        cv2.dilate(binary_img, kernel,binary_img, iterations=2)
        spec_mask = bin_thresholding(pupil_img, image_upper=highest_spike - spectral_offset)
        cv2.erode(spec_mask, kernel,spec_mask, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

        #open operation to remove eye lashes
        pupil_img = cv2.morphologyEx(pupil_img, cv2.MORPH_OPEN, kernel)

        if self.blur.value >1:
            pupil_img = cv2.medianBlur(pupil_img,self.blur.value)

        edges = cv2.Canny(pupil_img,
                            self.canny_thresh.value,
                            self.canny_thresh.value*self.canny_ratio.value,
                            apertureSize= self.canny_aperture.value)
        # edges = cv2.adaptiveThreshold(pupil_img,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.canny_aperture.value, 7)

        # remove edges in areas not dark enough and where the glint is (spectral refelction from IR leds)
        edges = cv2.min(edges, spec_mask)
        edges = cv2.min(edges,binary_img)

        if visualize:
            overlay =  img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]
            chn_img = grayscale(overlay)
            overlay[:,:,0] = cv2.max(chn_img,edges) #b channel
            # overlay[:,:,0] = cv2.max(chn_img,binary_img) #g channel
            overlay[:,:,2] = cv2.min(chn_img,spec_mask) #b channel

            pupil_img = frame.img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]
            # draw a frame around the automatic pupil ROI in overlay...
            pupil_img[::2,0] = 255,0,0
            pupil_img[::2,-1]= 255,0,0
            pupil_img[0,::2] = 255,0,0
            pupil_img[-1,::2]= 255,0,0

            pupil_img[::2,padding] = 255,0,0
            pupil_img[::2,-padding]= 255,0,0
            pupil_img[padding,::2] = 255,0,0
            pupil_img[-padding,::2]= 255,0,0

            frame.img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX] = pupil_img


        # from edges to contours to ellipses
        contours, hierarchy = cv2.findContours(edges,
                                            mode=cv2.RETR_LIST,
                                            method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
        # contours is a list containing array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )

        # the pupil target size is the one closest to the pupil_roi width or heigth (is the same)
        # self.target_size.value = p_roi.uX-p_roi.lX

        ### first we want to filter out the bad stuff
        # to short
        good_contours = [c for c in contours if c.shape[0]>self.min_contour_size]
        # now we learn things about each contour though looking at the curvature. For this we need to simplyfy the contour
        arprox_contours = [cv2.approxPolyDP(c,epsilon=2,closed=False) for c in good_contours]
        # cv2.drawContours(pupil_img,good_contours,-1,(255,255,0))
        # cv2.drawContours(pupil_img,arprox_contours,-1,(0,0,255))
        x_shift = 0 #just vor display
        color = zip(range(0,250,30),range(0,255,30)[::-1],range(230,250))
        split_contours = []
        for c in arprox_contours:
            curvature = GetAnglesPolyline(c)
            # print curvature
            # we split whenever there is a real kink (abs(curvature)<right angle) or a change in the genreal direction
            kink_idx = find_kink_and_dir_change(curvature,100)
            # kinks,k_index = convexity_defect(c,curvature)
            # print "kink_idx", kink_idx
            segs = split_at_corner_index(c,kink_idx)
            # print len(segs)
            # segs.sort(key=lambda e:-len(e))
            for s in segs:
                c = color.pop(0)
                color.append(c)
                cv2.polylines(pupil_img,[s],isClosed=False,color=c)
                split_contours.append(s)
                if visualize:
                    s = s.copy()
                    s[:,:,1] +=x_shift
                    s[:,:,0] +=img.shape[1]-coarse_pupil_width*2
                    x_shift +=5
                    cv2.polylines(img,[s],isClosed=False,color=c)
        # return {'timestamp':frame.timestamp,'norm_pupil':None}




        good_contours = [c for c in split_contours if c.shape[0]>5]
        # cv2.polylines(img,good_contours,isClosed=False,color=(255,255,0))
        shape = edges.shape
        ellipses = ((cv2.fitEllipse(c),c) for c in good_contours)
        ellipses = ((e,c) for e,c in ellipses if (padding < e[0][1] < shape[0]-padding and padding< e[0][0] < shape[1]-padding)) # center is close to roi center
        ellipses = ((e,c) for e,c in ellipses if binary_img[e[0][1],e[0][0]]) # center is on a dark pixel
        ellipses = [(size_deviation(e,self.target_size.value),e,c) for e,c in ellipses if is_round(e,self.target_ratio)] # roundness test
        result = []
        for size_dif,e,c in ellipses:
            pupil_ellipse = {}
            pupil_ellipse['contour'] = c
            a,b = e[1][0]/2.,e[1][1]/2. # majar minor radii of candidate ellipse
            # pupil_ellipse['circumference'] = np.pi*abs(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
            # pupil_ellipse['convex_hull'] = cv2.convexHull(pupil_ellipse['contour'])
            pupil_ellipse['contour_area'] = cv2.contourArea(cv2.convexHull(c))
            pupil_ellipse['ellipse_area'] = np.pi*a*b
            # print abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area'])
            if abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area']) <10:
                pupil_ellipse['goodness'] = 0 #perfect match we'll take this one
            else:
                pupil_ellipse['goodness'] = size_dif
            if visualize:
                    pass
                    # cv2.drawContours(pupil_img,[cv2.convexHull(c)],-1,(size_dif,size_dif,255))
                    # cv2.drawContours(pupil_img,[c],-1,(size_dif,size_dif,255))
            pupil_ellipse['center'] = u_roi.add_vector(p_roi.add_vector(e[0])) # compensate for roi offsets
            pupil_ellipse['angle'] = e[-1]
            pupil_ellipse['axes'] = e[1]
            pupil_ellipse['major'] = max(e[1])
            pupil_ellipse['minor'] = min(e[1])
            pupil_ellipse['ratio'] = pupil_ellipse['minor']/pupil_ellipse['major']
            pupil_ellipse['norm_pupil'] = normalize(pupil_ellipse['center'], (img.shape[1], img.shape[0]),flip_y=True )
            pupil_ellipse['timestamp'] = frame.timestamp
            result.append(pupil_ellipse)


        #### adding support
        if result:
            result.sort(key=lambda e: e['goodness'])
            # for now we assume that this contour is part of the pupil
            the_one = result[0]
            cv2.polylines(img,[the_one["contour"]],isClosed=False,color=(255,0,0),thickness=1)

            # but are there other segments that could be used for support?
            if len(result)>1:
                the_one = result[0]
                target_axes = the_one['axes'][0]
                # target_mean_curv = np.mean(curvature(the_one['contour'])
                new_support = [the_one['contour'],]
                for e in result[1:]:
                    manh_dist = abs(the_one["center"][0]-e['center'][0])+abs(the_one["center"][1]-e['center'][1])
                    size_dif = abs(the_one['major']-e['major'])
                    if manh_dist < the_one['major']:
                        if  size_dif < the_one['major']/3:
                            new_support.append(e["contour"])
                        else:
                            if visualize:
                                s = e["contour"].copy()
                                s[:,:,0] +=coarse_pupil_width*2
                                cv2.polylines(img,[s],isClosed=False,color=(0,0,255),thickness=1)
                        # cv2.polylines(img,[e["contour"]],isClosed=False,color=(0,100,255))

                new_support = np.concatenate(new_support)
                if visualize:
                    s = new_support.copy()
                    s[:,:,0] +=coarse_pupil_width*2
                    cv2.polylines(img,[s],isClosed=False,color=(255,100,255))
                self.goodness.value = the_one['goodness']

                ###### do the ellipse fit and filter think again
                ellipses = ((cv2.fitEllipse(c),c) for c in [new_support])
                ellipses = ((e,c) for e,c in ellipses if (padding < e[0][1] < shape[0]-padding and padding< e[0][0] < shape[1]-padding)) # center is close to roi center
                ellipses = ((e,c) for e,c in ellipses if binary_img[e[0][1],e[0][0]]) # center is on a dark pixel
                ellipses = [(size_deviation(e,self.target_size.value),e,c) for e,c in ellipses if is_round(e,self.target_ratio)] # roundness test
                for size_dif,e,c in ellipses:
                    pupil_ellipse = {}
                    pupil_ellipse['contour'] = c
                    a,b = e[1][0]/2.,e[1][1]/2. # majar minor radii of candidate ellipse
                    # pupil_ellipse['circumference'] = np.pi*abs(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
                    # pupil_ellipse['convex_hull'] = cv2.convexHull(pupil_ellipse['contour'])
                    pupil_ellipse['contour_area'] = cv2.contourArea(cv2.convexHull(c))
                    pupil_ellipse['ellipse_area'] = np.pi*a*b
                    # print abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area'])
                    if abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area']) <10:
                        pupil_ellipse['goodness'] = 0 #perfect match we'll take this one
                    else:
                        pupil_ellipse['goodness'] = size_dif
                    if visualize:
                            pass
                            # cv2.drawContours(pupil_img,[cv2.convexHull(c)],-1,(size_dif,size_dif,255))
                            # cv2.drawContours(pupil_img,[c],-1,(size_dif,size_dif,255))
                    pupil_ellipse['center'] = u_roi.add_vector(p_roi.add_vector(e[0])) # compensate for roi offsets
                    pupil_ellipse['angle'] = e[-1]
                    pupil_ellipse['axes'] = e[1]
                    pupil_ellipse['major'] = max(e[1])
                    pupil_ellipse['minor'] = min(e[1])
                    pupil_ellipse['ratio'] = pupil_ellipse['minor']/pupil_ellipse['major']
                    pupil_ellipse['norm_pupil'] = normalize(pupil_ellipse['center'], (img.shape[1], img.shape[0]),flip_y=True )
                    pupil_ellipse['timestamp'] = frame.timestamp
                    result = [pupil_ellipse,]
                # the_new_one = result[0]

            #done - if the new ellipse is good, we just overwrote the old result

            # update the target size
            if result[0]['goodness'] ==0: # perfect match!
                self.target_size.value = result[0]['major']
            else:
                self.target_size.value  = self.target_size.value +  .2 * (result[0]['major']-self.target_size.value)
                result.sort(key=lambda e: abs(e['major']-self.target_size.value))
            if visualize:
                pass
            return result[0]

        else:
            self.goodness.value = 100
            no_result = {}
            no_result['timestamp'] = frame.timestamp
            no_result['norm_pupil'] = None
            return no_result



    def create_atb_bar(self,pos):
        self._bar = atb.Bar(name = "Canny_Pupil_Detector", label="Pupil_Detector",
            help="pupil detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 100))
        self._bar.add_var("pupil_intensity_range",self.intensity_range)
        self._bar.add_var("Pupil_Aparent_Size",self.target_size)
        self._bar.add_var("Pupil_Shade",self.bin_thresh, readonly=True)
        self._bar.add_var("Pupil_Certainty",self.goodness, readonly=True)
        self._bar.add_var("Image_Blur",self.blur, step=2,min=1,max=9)
        self._bar.add_var("Canny_aparture",self.canny_aperture, step=2,min=3,max=7)
        self._bar.add_var("canny_threshold",self.canny_thresh, step=1,min=0)
        self._bar.add_var("Canny_ratio",self.canny_ratio, step=1,min=1)