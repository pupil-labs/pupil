
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
from file_methods import Persistent_Dict
import numpy as np
from methods import *
import atb
from ctypes import c_int,c_bool,c_float
from c_methods import eye_filter
from glfw import *
from gl_utils import  draw_gl_texture,adjust_gl_view, clear_gl_screen, draw_gl_point_norm, draw_gl_polyline,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
from template import Pupil_Detector


import logging
logger = logging.getLogger(__name__)


# try:
#     np.euler_gamma #constant introduced with 1.8, numpy.__version__ will give you strings with non int chars...
# except AttributeError as error:
#     logger.error("This module requires numpy 1.8 or greater. Please upgrade your version of numpy.")
#     raise error


class Canny_Detector(Pupil_Detector):
    """a Pupil detector based on Canny_Edges"""
    def __init__(self,g_pool):
        super(Canny_Detector, self).__init__()

        # load session persistent settings
        self.session_settings =Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_detector') )

        # coase pupil filter params
        self.coarse_detection = c_bool(self.load('coarse_detection',True))
        self.coarse_filter_min = 100
        self.coarse_filter_max = 400

        # canny edge detection params
        self.blur = 1
        self.canny_thresh = 159
        self.canny_ratio= 2
        self.canny_aperture = 5

        # edge intensity filter params
        self.intensity_range = c_int(self.load('intensity_range',11))
        self.bin_thresh = c_int(0)

        # contour prefilter params
        self.min_contour_size = c_int(self.load('min_contour_size',80))

        #ellipse filter params
        self.inital_ellipse_fit_threshhold = 1.8
        self.min_ratio = .3
        self.pupil_min = c_float(self.load('pupil_min',40.))
        self.pupil_max = c_float(self.load('pupil_max',150.))
        self.target_size= c_float(100.)
        self.strong_perimeter_ratio_range = .8, 1.1
        self.strong_area_ratio_range = .6,1.1
        self.final_perimeter_ratio_range = self.load("final_perimeter_ratio_range",[.6, 1.2])
        self.strong_prior = None


        #detector dignostics
        #confidance in the mesurement 0(bad) to 1 (perfect)
        # in this case we take the support ratio capped at 1. (uncapped if the pupil comes from prior)
        self.confidence = c_float(0)
        self.confidence_hist = []


        #debug window
        self.suggested_size = 640,480
        self._window = None
        self.window_should_open = False
        self.window_should_close = False

        #debug settings
        self.should_sleep = False

    def load(self, var_name, default):
        return self.session_settings.get(var_name,default)
    def save(self, var_name, var):
            self.session_settings[var_name] = var

    def detect(self,frame,user_roi,visualize=False):
        u_r = user_roi
        if self.window_should_open:
            self.open_window((frame.img.shape[1],frame.img.shape[0]))
        if self.window_should_close:
            self.close_window()

        if self._window:
            debug_img = np.zeros(frame.img.shape,frame.img.dtype)


        #get the user_roi
        img = frame.img
        r_img = img[u_r.view]
        gray_img = cv2.cvtColor(r_img,cv2.COLOR_BGR2GRAY)


        # coarse pupil detection

        if self.coarse_detection.value:
            integral = cv2.integral(gray_img)
            integral =  np.array(integral,dtype=c_float)
            x,y,w,response = eye_filter(integral,self.coarse_filter_min,self.coarse_filter_max)
            p_r = Roi(gray_img.shape)
            if w>0:
                p_r.set((y,x,y+w,x+w))
            else:
                p_r.set((0,0,-1,-1))
        else:
            p_r = Roi(gray_img.shape)
            p_r.set((0,0,None,None))
            w = img.shape[0]/2

        coarse_pupil_width = w/2.
        padding = coarse_pupil_width/4.
        pupil_img = gray_img[p_r.view]



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

        if self.blur > 1:
            pupil_img = cv2.medianBlur(pupil_img,self.blur.value)

        edges = cv2.Canny(pupil_img,
                            self.canny_thresh,
                            self.canny_thresh*self.canny_ratio,
                            apertureSize= self.canny_aperture)


        # remove edges in areas not dark enough and where the glint is (spectral refelction from IR leds)
        edges = cv2.min(edges, spec_mask)
        edges = cv2.min(edges,binary_img)

        overlay =  img[u_r.view][p_r.view]
        if visualize:
            b,g,r = overlay[:,:,0],overlay[:,:,1],overlay[:,:,2]
            g[:] = cv2.max(g,edges)
            b[:] = cv2.max(b,binary_img)
            b[:] = cv2.min(b,spec_mask)

            # draw a frame around the automatic pupil ROI in overlay.
            overlay[::2,0] = 255 #yeay numpy broadcasting
            overlay[::2,-1]= 255
            overlay[0,::2] = 255
            overlay[-1,::2]= 255
            # draw a frame around the area we require the pupil center to be.
            overlay[padding:-padding:4,padding] = 255
            overlay[padding:-padding:4,-padding]= 255
            overlay[padding,padding:-padding:4] = 255
            overlay[-padding,padding:-padding:4]= 255

        if visualize:
            c = (100.,frame.img.shape[0]-100.)
            e_max = ((c),(self.pupil_max.value,self.pupil_max.value),0)
            e_recent = ((c),(self.target_size.value,self.target_size.value),0)
            e_min = ((c),(self.pupil_min.value,self.pupil_min.value),0)
            cv2.ellipse(frame.img,e_min,(0,0,255),1)
            cv2.ellipse(frame.img,e_recent,(0,255,0),1)
            cv2.ellipse(frame.img,e_max,(0,0,255),1)

        #get raw edge pix for later
        raw_edges = cv2.findNonZero(edges)

        def ellipse_true_support(e,raw_edges):
            a,b = e[1][0]/2.,e[1][1]/2. # major minor radii of candidate ellipse
            ellipse_circumference = np.pi*abs(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
            distances = dist_pts_ellipse(e,raw_edges)
            support_pixels = raw_edges[distances<=1.3]
            # support_ratio = support_pixel.shape[0]/ellipse_circumference
            return support_pixels,ellipse_circumference

        # if we had a good ellipse before ,let see if it is still a good first guess:
        if self.strong_prior:
            e = p_r.sub_vector(u_r.sub_vector(self.strong_prior[0])),self.strong_prior[1],self.strong_prior[2]

            self.strong_prior = None
            if raw_edges is not None:
                support_pixels,ellipse_circumference = ellipse_true_support(e,raw_edges)
                support_ratio =  support_pixels.shape[0]/ellipse_circumference
                if support_ratio >= self.strong_perimeter_ratio_range[0]:
                    refit_e = cv2.fitEllipse(support_pixels)
                    if self._window:
                        cv2.ellipse(debug_img,e,(255,100,100),thickness=4)
                        cv2.ellipse(debug_img,refit_e,(0,0,255),thickness=1)
                    e = refit_e
                    self.strong_prior = u_r.add_vector(p_r.add_vector(e[0])),e[1],e[2]
                    goodness = min(1.,support_ratio)
                    pupil_ellipse = {}
                    pupil_ellipse['confidence'] = goodness
                    pupil_ellipse['ellipse'] = e
                    pupil_ellipse['roi_center'] = e[0]
                    pupil_ellipse['major'] = max(e[1])
                    pupil_ellipse['minor'] = min(e[1])
                    pupil_ellipse['apparent_pupil_size'] = max(e[1])
                    pupil_ellipse['axes'] = e[1]
                    pupil_ellipse['angle'] = e[2]
                    e_img_center =u_r.add_vector(p_r.add_vector(e[0]))
                    norm_center = normalize(e_img_center,(frame.img.shape[1], frame.img.shape[0]),flip_y=True)
                    pupil_ellipse['norm_pupil'] = norm_center
                    pupil_ellipse['center'] = e_img_center
                    pupil_ellipse['timestamp'] = frame.timestamp

                    self.target_size.value = max(e[1])

                    self.confidence.value = goodness
                    self.confidence_hist.append(goodness)
                    self.confidence_hist[:-200]=[]
                    if self._window:
                        #draw a little animation of confidence
                        cv2.putText(debug_img, 'good',(410,debug_img.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,100,100))
                        cv2.putText(debug_img, 'threshold',(410,debug_img.shape[0]-int(self.final_perimeter_ratio_range[0]*100)), cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,100,100))
                        cv2.putText(debug_img, 'no detection',(410,debug_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,100,100))
                        lines = np.array([[[2*x,debug_img.shape[0]-int(100*y)],[2*x,debug_img.shape[0]]] for x,y in enumerate(self.confidence_hist)])
                        cv2.polylines(debug_img,lines,isClosed=False,color=(255,100,100))
                        self.gl_display_in_window(debug_img)
                    return pupil_ellipse





        # from edges to contours
        contours, hierarchy = cv2.findContours(edges,
                                            mode=cv2.RETR_LIST,
                                            method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
        # contours is a list containing array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )

        ### first we want to filter out the bad stuff
        # to short
        good_contours = [c for c in contours if c.shape[0]>self.min_contour_size.value]
        # now we learn things about each contour through looking at the curvature.
        # For this we need to simplyfy the contour so that pt to pt angles become more meaningfull
        aprox_contours = [cv2.approxPolyDP(c,epsilon=1.5,closed=False) for c in good_contours]

        if self._window:
            x_shift = coarse_pupil_width*2
            color = zip(range(0,250,15),range(0,255,15)[::-1],range(230,250))
        split_contours = []
        for c in aprox_contours:
            curvature = GetAnglesPolyline(c)
            # we split whenever there is a real kink (abs(curvature)<right angle) or a change in the genreal direction
            kink_idx = find_kink_and_dir_change(curvature,80)
            segs = split_at_corner_index(c,kink_idx)

            #TODO: split at shart inward turns
            for s in segs:
                if s.shape[0]>2:
                    split_contours.append(s)
                    if self._window:
                        c = color.pop(0)
                        color.append(c)
                        s = s.copy()
                        s[:,:,0] += debug_img.shape[1]-coarse_pupil_width*2
                        # s[:,:,0] += x_shift
                        # x_shift += 5
                        cv2.polylines(debug_img,[s],isClosed=False,color=map(lambda x: x,c),thickness = 1,lineType=4)#cv2.CV_AA

        split_contours.sort(key=lambda x:-x.shape[0])
        # print [x.shape[0]for x in split_contours]
        if len(split_contours) == 0:
            # not a single usefull segment found -> no pupil found
            self.confidence.value = 0
            self.confidence_hist.append(0)
            if self._window:
                self.gl_display_in_window(debug_img)
            return {'timestamp':frame.timestamp,'norm_pupil':None}


        # removing stubs makes combinatorial search feasable
        split_contours = [c for c in split_contours if c.shape[0]>3]

        def ellipse_filter(e):
            in_center = padding < e[0][1] < pupil_img.shape[0]-padding and padding < e[0][0] < pupil_img.shape[1]-padding
            if in_center:
                is_round = min(e[1])/max(e[1]) >= self.min_ratio
                if is_round:
                    right_size = self.pupil_min.value <= max(e[1]) <= self.pupil_max.value
                    if right_size:
                        return True
            return False

        def ellipse_on_blue(e):
            center_on_dark = binary_img[e[0][1],e[0][0]]
            return bool(center_on_dark)

        def ellipse_support_ratio(e,contours):
            a,b = e[1][0]/2.,e[1][1]/2. # major minor radii of candidate ellipse
            ellipse_area =  np.pi*a*b
            ellipse_circumference = np.pi*abs(3*(a+b)-np.sqrt(10*a*b+3*(a**2+b**2)))
            actual_area = cv2.contourArea(cv2.convexHull(np.concatenate(contours)))
            actual_contour_length = sum([cv2.arcLength(c,closed=False) for c in contours])
            area_ratio = actual_area / ellipse_area
            perimeter_ratio = actual_contour_length / ellipse_circumference #we assume here that the contour lies close to the ellipse boundary
            return perimeter_ratio,area_ratio


        def final_fitting(c,edges):
            #use the real edge pixels to fit, not the aproximated contours
            support_mask = np.zeros(edges.shape,edges.dtype)
            cv2.polylines(support_mask,c,isClosed=False,color=(255,255,255),thickness=2)
            # #draw into the suport mast with thickness 2
            new_edges = cv2.min(edges, support_mask)
            new_contours = cv2.findNonZero(new_edges)
            if self._window:
                new_edges[new_edges!=0] = 255
                overlay[:,:,1] = cv2.max(overlay[:,:,1], new_edges)
                overlay[:,:,2] = cv2.max(overlay[:,:,2], new_edges)
                overlay[:,:,2] = cv2.max(overlay[:,:,2], new_edges)
            new_e = cv2.fitEllipse(new_contours)
            return new_e,new_contours


        # finding poential candidates for ellipse seeds that describe the pupil.
        strong_seed_contours = []
        weak_seed_contours = []
        for idx, c in enumerate(split_contours):
            if c.shape[0] >=5:
                e = cv2.fitEllipse(c)
                # is this ellipse a plausible canditate for a pupil?
                if ellipse_filter(e):
                    distances = dist_pts_ellipse(e,c)
                    fit_variance = np.sum(distances**2)/float(distances.shape[0])
                    if fit_variance <= self.inital_ellipse_fit_threshhold:
                        # how much ellipse is supported by this contour?
                        perimeter_ratio,area_ratio = ellipse_support_ratio(e,[c])
                        # logger.debug('Ellipse no %s with perimeter_ratio: %s , area_ratio: %s'%(idx,perimeter_ratio,area_ratio))
                        if self.strong_perimeter_ratio_range[0]<= perimeter_ratio <= self.strong_perimeter_ratio_range[1] and self.strong_area_ratio_range[0]<= area_ratio <= self.strong_area_ratio_range[1]:
                            strong_seed_contours.append(idx)
                            if self._window:
                                cv2.polylines(debug_img,[c],isClosed=False,color=(255,100,100),thickness=4)
                                e = (e[0][0]+debug_img.shape[1]-coarse_pupil_width*4,e[0][1]),e[1],e[2]
                                cv2.ellipse(debug_img,e,color=(255,100,100),thickness=3)
                        else:
                            weak_seed_contours.append(idx)
                            if self._window:
                                cv2.polylines(debug_img,[c],isClosed=False,color=(255,0,0),thickness=2)
                                e = (e[0][0]+debug_img.shape[1]-coarse_pupil_width*4,e[0][1]),e[1],e[2]
                                cv2.ellipse(debug_img,e,color=(255,0,0))

        sc = np.array(split_contours)


        if strong_seed_contours:
            seed_idx = strong_seed_contours
        elif weak_seed_contours:
            seed_idx = weak_seed_contours

        if not (strong_seed_contours or weak_seed_contours):
            if self._window:
                self.gl_display_in_window(debug_img)
            self.confidence.value = 0
            self.confidence_hist.append(0)
            return {'timestamp':frame.timestamp,'norm_pupil':None}

        # if self._window:
        #     cv2.polylines(debug_img,[split_contours[i] for i in seed_idx],isClosed=False,color=(255,255,100),thickness=3)

        def ellipse_eval(contours):
            c = np.concatenate(contours)
            e = cv2.fitEllipse(c)
            d = dist_pts_ellipse(e,c)
            fit_variance = np.sum(d**2)/float(d.shape[0])
            return fit_variance <= self.inital_ellipse_fit_threshhold


        solutions = pruning_quick_combine(split_contours,ellipse_eval,seed_idx,max_evals=1000,max_depth=5)
        solutions = filter_subsets(solutions)
        ratings = []


        for s in solutions:
            e = cv2.fitEllipse(np.concatenate(sc[s]))
            if self._window:
                cv2.ellipse(debug_img,e,(0,150,100))
            support_pixels,ellipse_circumference = ellipse_true_support(e,raw_edges)
            support_ratio =  support_pixels.shape[0]/ellipse_circumference
            # TODO: refine the selection of final canditate
            if support_ratio >=self.final_perimeter_ratio_range[0] and ellipse_filter(e):
                ratings.append(support_pixels.shape[0])
                if support_ratio >=self.strong_perimeter_ratio_range[0]:
                    self.strong_prior = u_r.add_vector(p_r.add_vector(e[0])),e[1],e[2]
                    if self._window:
                        cv2.ellipse(debug_img,e,(0,255,255),thickness = 2)
            else:
                #not a valid solution, bad rating
                ratings.append(-1)


        # selected ellipse
        if max(ratings) == -1:
            #no good final ellipse found
            if self._window:
                self.gl_display_in_window(debug_img)
            self.confidence.value = 0
            self.confidence_hist.append(0)
            return {'timestamp':frame.timestamp,'norm_pupil':None}

        best = solutions[ratings.index(max(ratings))]
        e = cv2.fitEllipse(np.concatenate(sc[best]))

        #final calculation of goodness of fit
        support_pixels,ellipse_circumference = ellipse_true_support(e,raw_edges)
        support_ratio =  support_pixels.shape[0]/ellipse_circumference
        goodness = min(1.,support_ratio)

        #final fitting and return of result
        new_e,final_edges = final_fitting(sc[best],edges)
        size_dif = abs(1 - max(e[1])/max(new_e[1]))
        if ellipse_filter(new_e) and size_dif < .3:
            if self._window:
                cv2.ellipse(debug_img,new_e,(0,255,0))
            e = new_e


        pupil_ellipse = {}
        pupil_ellipse['confidence'] = goodness
        pupil_ellipse['ellipse'] = e
        pupil_ellipse['pos_in_roi'] = e[0]
        pupil_ellipse['major'] = max(e[1])
        pupil_ellipse['apparent_pupil_size'] = max(e[1])
        pupil_ellipse['minor'] = min(e[1])
        pupil_ellipse['axes'] = e[1]
        pupil_ellipse['angle'] = e[2]
        e_img_center =u_r.add_vector(p_r.add_vector(e[0]))
        norm_center = normalize(e_img_center,(frame.img.shape[1], frame.img.shape[0]),flip_y=True)
        pupil_ellipse['norm_pupil'] = norm_center
        pupil_ellipse['center'] = e_img_center
        pupil_ellipse['timestamp'] = frame.timestamp

        self.target_size.value = max(e[1])

        self.confidence.value = goodness
        self.confidence_hist.append(goodness)
        self.confidence_hist[:-200]=[]
        if self._window:
            #draw a little animation of confidence
            cv2.putText(debug_img, 'good',(410,debug_img.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,100,100))
            cv2.putText(debug_img, 'threshold',(410,debug_img.shape[0]-int(self.final_perimeter_ratio_range[0]*100)), cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,100,100))
            cv2.putText(debug_img, 'no detection',(410,debug_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,100,100))
            lines = np.array([[[2*x,debug_img.shape[0]-int(100*y)],[2*x,debug_img.shape[0]]] for x,y in enumerate(self.confidence_hist)])
            cv2.polylines(debug_img,lines,isClosed=False,color=(255,100,100))
            self.gl_display_in_window(debug_img)
        return pupil_ellipse




    # Display and interface methods
    def set_final_perimeter_ratio_range(self,val):
        self.final_perimeter_ratio_range[0] = val

    def create_atb_bar(self,pos):
        self._bar = atb.Bar(name = "Canny_Pupil_Detector", label="Pupil_Detector",
            help="pupil detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=pos,refresh=.3, size=(200, 100))
        self._bar.add_var("use coarse detection",self.coarse_detection,help="Disbale when you have trouble with detection when using Mascara.")
        self._bar.add_button("open debug window", self.toggle_window,help="Open a debug window that shows geeky visual feedback from the algorithm.")
        self._bar.add_var("pupil_intensity_range",self.intensity_range,help="Using alorithm view set this as low as possible but so that the pupil is always fully overlayed in blue.")
        self._bar.add_var("pupil_min",self.pupil_min,min=1,help="Setting good bounds will increase detection robustness. Use alorithm view to see.")
        self._bar.add_var("pupil_max",self.pupil_max,min=1,help="Setting good bounds will increase detection robustness. Use alorithm view to see.")
        self._bar.add_var("Pupil_Aparent_Size",self.target_size,readonly=True)
        self._bar.add_var("Contour min length",self.min_contour_size,help="Setting this low will make the alorithm try to connect even smaller arcs to find the pupil but cost you cpu time!")
        self._bar.add_var("confidece threshold",c_float(0),getter= lambda: self.final_perimeter_ratio_range[0], setter=self.set_final_perimeter_ratio_range,step=.05,min=0.,max=1. ,
                            help="Fraction of pupil boundry that has to be visible and detected for the resukt to be declared valid.")
        # self._bar.add_var("Pupil_Shade",self.bin_thresh, readonly=True)
        self._bar.add_var("confidence",self.confidence, readonly=True,help="The measure of confidence is a number between 0 and 1 of how sure the algorithm is about the detected pupil. We currenlty use the fraction of pupil boundry edge that is used as support for the ellipse result.")
        # self._bar.add_var("Image_Blur",self.blur, step=2,min=1,max=9)
        # self._bar.add_var("Canny_aparture",self.canny_aperture, step=2,min=3,max=7)
        # self._bar.add_var("canny_threshold",self.canny_thresh, step=1,min=0)
        # self._bar.add_var("Canny_ratio",self.canny_ratio, step=1,min=1)

    def toggle_window(self):
        if self._window:
            self.window_should_close = True
        else:
            self.window_should_open = True

    def open_window(self,size):
        if not self._window:
            if 0: #we are not fullscreening
                monitor = glfwGetMonitors()[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= size

            active_window = glfwGetCurrentContext()
            self._window = glfwCreateWindow(height, width, "Plugin Window", monitor=monitor, share=active_window)
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

            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

            self.window_should_open = False

    # window calbacks
    def on_resize(self,window,w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h,window)
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
        make_coord_system_norm_based()
        draw_gl_texture(img,interpolation=False)
        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def cleanup(self):
        self.save('intensity_range',self.intensity_range.value)
        self.save('pupil_min',self.pupil_min.value)
        self.save('pupil_max',self.pupil_max.value)
        self.save('min_contour_size',self.min_contour_size.value)
        self.save('final_perimeter_ratio_range',self.final_perimeter_ratio_range)
        self.session_settings.close()
