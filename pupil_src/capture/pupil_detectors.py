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
import random
from glfw import *
from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen, draw_gl_point_norm, draw_gl_polyline,basic_gl_setup


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


class Canny_Detector(Pupil_Detector):
    """a Pupil detector based on Canny_Edges"""
    def __init__(self):
        super(Canny_Detector, self).__init__()

        # coase pupil filter params
        self.coarse_filter_min = 100
        self.coarse_filter_max = 400

        # canny edge detection params
        self.blur = c_int(1)
        self.canny_thresh = c_int(200)
        self.canny_ratio= c_int(2)
        self.canny_aperture = c_int(7)

        # edge intensity filter params
        self.intensity_range = c_int(17)
        self.bin_thresh = c_int(0)

        # contour prefilter params
        self.min_contour_size = 80

        #ellipse filter params
        self.target_ratio=1.0
        self.target_size=c_float(100.)
        self.goodness = c_float(1.)
        self.size_tolerance=10.


        #debug window
        self._window = None
        self.window_should_open = False
        self.window_should_close = False

        #debug settings
        self.should_sleep = False

    def detect(self,frame,u_roi,visualize=False):

        if self.window_should_open:
            self.open_window()
        if self.window_should_close:
            self.close_window()



        #get the user_roi
        img = frame.img
        r_img = img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX]
        gray_img = grayscale(r_img)


        # coarse pupil detection
        integral = cv2.integral(gray_img)
        integral =  np.array(integral,dtype=c_float)
        x,y,w,response = eye_filter(integral,self.coarse_filter_min,self.coarse_filter_max)
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
            overlay[:,:,2] = cv2.max(chn_img,edges) #b channel
            overlay[:,:,0] = cv2.max(chn_img,binary_img) #g channel
            overlay[:,:,1] = cv2.min(chn_img,spec_mask) #b channel

            pupil_img = frame.img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX]
            # draw a frame around the automatic pupil ROI in overlay...
            pupil_img[::2,0] = 255,255,255
            pupil_img[::2,-1]= 255,255,255
            pupil_img[0,::2] = 255,255,255
            pupil_img[-1,::2]= 255,255,255

            pupil_img[::2,padding] = 255,255,255
            pupil_img[::2,-padding]= 255,255,255
            pupil_img[padding,::2] = 255,255,255
            pupil_img[-padding,::2]= 255,255,255

            frame.img[u_roi.lY:u_roi.uY,u_roi.lX:u_roi.uX][p_roi.lY:p_roi.uY,p_roi.lX:p_roi.uX] = pupil_img


        # from edges to contours
        contours, hierarchy = cv2.findContours(edges,
                                            mode=cv2.RETR_LIST,
                                            method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
        # contours is a list containing array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )


        ### first we want to filter out the bad stuff
        # to short
        good_contours = [c for c in contours if c.shape[0]>self.min_contour_size]
        # now we learn things about each contour though looking at the curvature. For this we need to simplyfy the contour
        arprox_contours = [cv2.approxPolyDP(c,epsilon=1.5,closed=False) for c in good_contours]
        # cv2.drawContours(pupil_img,good_contours,-1,(255,255,0))
        # cv2.drawContours(pupil_img,arprox_contours,-1,(0,0,255))

        if self._window:
            debug_img = np.zeros(img.shape,img.dtype)

        x_shift = coarse_pupil_width*2 #just vor display
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
                split_contours.append(s)
                if self._window:
                    c = color.pop(0)
                    color.append(c)
                    # if s.shape[0] >=5:
                    #     cv2.polylines(debug_img,[s],isClosed=False,color=c)
                    s = s.copy()
                    s[:,:,1] +=  coarse_pupil_width*2
                    cv2.polylines(debug_img,[s],isClosed=False,color=c)
                    s[:,:,0] += x_shift
                    x_shift += 5
                    cv2.polylines(debug_img,[s],isClosed=False,color=c)
        # return {'timestamp':frame.timestamp,'norm_pupil':None}

        #these segments may now be smaller, we need to get rid of those not long enough for ellipse fitting
        good_contours = [c for c in split_contours if c.shape[0]>=5]
        # cv2.polylines(img,good_contours,isClosed=False,color=(255,255,0))

        shape = edges.shape
        ellipses = ((cv2.fitEllipse(c),c) for c in good_contours)
        ellipses = ((e,c) for e,c in ellipses if (padding < e[0][1] < shape[0]-padding and padding< e[0][0] < shape[1]-padding)) # center is close to roi center
        ellipses = ((e,c) for e,c in ellipses if binary_img[e[0][1],e[0][0]]) # center is on a dark pixel
        ellipses = [(e,c) for e,c in ellipses if is_round(e,self.target_ratio)] # roundness test
        result = []
        for e,c in ellipses:
            size_dif = size_deviation(e,self.target_size.value)
            pupil_ellipse = {}
            pupil_ellipse['contour'] = c
            a,b = e[1][0]/2.,e[1][1]/2. # majar minor radii of candidate ellipse
            pupil_ellipse['circumference'] = np.pi*abs(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
            # pupil_ellipse['convex_hull'] = cv2.convexHull(pupil_ellipse['contour'])
            pupil_ellipse['contour_area'] = cv2.contourArea(cv2.convexHull(c))
            pupil_ellipse['ellipse_area'] = np.pi*a*b
            # print abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area'])
            if abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area']) <10:
                pupil_ellipse['goodness'] = abs(pupil_ellipse['contour_area']-pupil_ellipse['ellipse_area'])/10 #perfect match we'll take this one
            else:
                pupil_ellipse['goodness'] = size_dif
            if visualize:
                    pass
                    # cv2.drawContours(pupil_img,[cv2.convexHull(c)],-1,(size_dif,size_dif,255))
                    # cv2.drawContours(pupil_img,[c],-1,(size_dif,size_dif,255))
            pupil_ellipse['pupil_center'] = e[0] # compensate for roi offsets
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
            # (center, size, angle) = cv2.fitEllipse(the_one['contour'])
            # print "itself"
            distances =  dist_pts_ellipse(cv2.fitEllipse(the_one['contour']),the_one['contour'])
            # print np.average(distances)
            # print np.sum(distances)/float(distances.shape[0])
            # print "other"
            # if self._window:
                # cv2.polylines(debug_img,[result[-1]['contour']],isClosed=False,color=(255,255,255),thickness=3)
            with_another = np.concatenate((result[-1]['contour'],the_one['contour']))
            distances =  dist_pts_ellipse(cv2.fitEllipse(with_another),with_another)
            # if 1.5 > np.sum(distances)/float(distances.shape[0]):
            #     if self._window:
            #         cv2.polylines(debug_img,[result[-1]['contour']],isClosed=False,color=(255,255,255),thickness=3)

            perimeter_ratio =  cv2.arcLength(the_one["contour"],closed=False)/the_one['circumference']
            if perimeter_ratio > .9:
                size_thresh = 0
                eccentricity_thresh = 0
            elif perimeter_ratio > .5:
                size_thresh = the_one['major']/(5.)
                eccentricity_thresh = the_one['major']/2.
                self.should_sleep = True
            else:
                size_thresh = the_one['major']/(3.)
                eccentricity_thresh = the_one['major']/2.
                self.should_sleep = True
            if self._window:
                center = np.uint16(np.around(the_one['pupil_center']))
                cv2.circle(debug_img,tuple(center),int(eccentricity_thresh),(0,255,0),1)

            if self._window:
                cv2.polylines(debug_img,[the_one["contour"]],isClosed=False,color=(255,0,0),thickness=2)
                s = the_one["contour"].copy()
                s[:,:,0] +=coarse_pupil_width*2
                cv2.polylines(debug_img,[s],isClosed=False,color=(255,0,0),thickness=2)
            # but are there other segments that could be used for support?
            new_support = [the_one['contour'],]
            if len(result)>1:
                the_one = result[0]
                target_axes = the_one['axes'][0]
                # target_mean_curv = np.mean(curvature(the_one['contour'])
                for e in result:

                    # with_another = np.concatenate((e['contour'],the_one['contour']))
                    # with_another = np.concatenate([r['contour'] for r in result])
                    with_another = e['contour']
                    distances =  dist_pts_ellipse(cv2.fitEllipse(with_another),with_another)
                    # print np.std(distances)
                    thick =  int(np.std(distances))
                    if 1.5 > np.average(distances) or 1:
                        if self._window:
                            # print thick
                            thick = min(20,thick)
                            cv2.polylines(debug_img,[e['contour']],isClosed=False,color=(255,255,255),thickness=thick)

                    if self._window:
                        cv2.polylines(debug_img,[e["contour"]],isClosed=False,color=(0,100,100))
                    center_dist = cv2.arcLength(np.array([the_one["pupil_center"],e['pupil_center']],dtype=np.int32),closed=False)
                    size_dif = abs(the_one['major']-e['major'])

                    # #lets make sure the countour is not behind the_one/'s coutour
                    # center_point = np.uint16(np.around(the_one['pupil_center']))
                    # other_center_point = np.uint16(np.around(e['pupil_center']))

                    # mid_point =  the_one["contour"][the_one["contour"].shape[0]/2][0]
                    # other_mid_point =  e["contour"][e["contour"].shape[0]/2][0]

                    # #reflect around mid_point
                    # p = center_point - mid_point
                    # p = np.array((-p[1],-p[0]))
                    # mir_center_point = p + mid_point
                    # dist_mid = cv2.arcLength(np.array([mid_point,other_mid_point]),closed=False)
                    # dist_center = cv2.arcLength(np.array([center_point,other_mid_point]),closed=False)
                    # if self._window:
                    #     cv2.circle(debug_img,tuple(center_point),3,(0,255,0),2)
                    #     cv2.circle(debug_img,tuple(other_center_point),2,(0,0,255),1)
                    #     # cv2.circle(debug_img,tuple(mir_center_point),3,(0,255,0),2)
                    #     # cv2.circle(debug_img,tuple(mid_point),2,(0,255,0),1)
                    #     # cv2.circle(debug_img,tuple(other_mid_point),2,(0,0,255),1)
                    #     cv2.polylines(debug_img,[np.array([center_point,other_mid_point]),np.array([mid_point,other_mid_point])],isClosed=False,color=(0,255,0))


                    if center_dist < eccentricity_thresh:
                    # print dist_mid-dist_center
                    # if dist_mid > dist_center-20:

                        if  size_dif < size_thresh:


                            new_support.append(e["contour"])
                            if self._window:
                                cv2.polylines(debug_img,[s],isClosed=False,color=(255,0,0),thickness=1)
                                s = e["contour"].copy()
                                s[:,:,0] +=coarse_pupil_width*2
                                cv2.polylines(debug_img,[s],isClosed=False,color=(255,255,0),thickness=1)

                        else:
                            if self._window:
                                s = e["contour"].copy()
                                s[:,:,0] +=coarse_pupil_width*2
                                cv2.polylines(debug_img,[s],isClosed=False,color=(0,0,255),thickness=1)
                    else:
                        if self._window:
                            cv2.polylines(debug_img,[s],isClosed=False,color=(0,255,255),thickness=1)

                    # new_support = np.concatenate(new_support)

            self.goodness.value = the_one['goodness']

            ###here we should AND original mask, selected contours with 2px thinkness (and 2px fitted ellipse -is the last one a good idea??)
            support_mask = np.zeros(edges.shape,edges.dtype)
            cv2.polylines(support_mask,new_support,isClosed=False,color=(255,255,255),thickness=2)
            # #draw into the suport mast with thickness 2
            new_edges = cv2.min(edges, support_mask)
            new_contours = cv2.findNonZero(new_edges)
            if self._window:
                debug_img[0:support_mask.shape[0],0:support_mask.shape[1],2] = new_edges


            ###### do the ellipse fit and filter think again
            ellipses = ((cv2.fitEllipse(c),c) for c in [new_contours])
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

        if self._window:
            self.gl_display_in_window(debug_img)
            if self.should_sleep:
                # sleep(3)
                self.should_sleep = False
        if result:
            # update the target size
            if result[0]['goodness'] >=3: # perfect match!
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
        self._bar.add_button("open debug window", self.toggle_window)
        self._bar.add_var("pupil_intensity_range",self.intensity_range)
        self._bar.add_var("Pupil_Aparent_Size",self.target_size)
        self._bar.add_var("Pupil_Shade",self.bin_thresh, readonly=True)
        self._bar.add_var("Pupil_Certainty",self.goodness, readonly=True)
        self._bar.add_var("Image_Blur",self.blur, step=2,min=1,max=9)
        self._bar.add_var("Canny_aparture",self.canny_aperture, step=2,min=3,max=7)
        self._bar.add_var("canny_threshold",self.canny_thresh, step=1,min=0)
        self._bar.add_var("Canny_ratio",self.canny_ratio, step=1,min=1)

    def toggle_window(self):
        if self._window:
            self.window_should_close = True
        else:
            self.window_should_open = True

    def open_window(self):
        if not self._window:
            if 0: #we are not fullscreening
                monitor = self.monitor_handles[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 640,360

            active_window = glfwGetCurrentContext()
            self._window = glfwCreateWindow(height, width, "Plugin Window", monitor=monitor, share=None)
            if not 0:
                glfwSetWindowPos(self._window,200,0)

            self.on_resize(self._window,height,width)

            #Register callbacks
            glfwSetWindowSizeCallback(self._window,self.on_resize)
            # glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            glfwMakeContextCurrent(active_window)

            self.window_should_open = False

    # window calbacks
    def on_resize(self,window,w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfwMakeContextCurrent(active_window)

    def on_close(self,window):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False


    def gl_display_in_window(self,img):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)
        clear_gl_screen()
        # gl stuff that will show on your plugin window goes here
        draw_gl_texture(img,interpolation=False)
        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)



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

