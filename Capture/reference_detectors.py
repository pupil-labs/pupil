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
from methods import normalize,denormalize
from c_methods import ring_filter,c_float,c_bool,c_int

from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_polyline

import atb

import audio

class Ref_Detector_Template(object):
    """
    template of reference detectors class
    build a detector based on this class.

    Your derived class needs to have interfaces
    defined by these methods:
    you NEED to do at least what is done in these fn-prototypes

    ...
    """
    def __init__(self,global_calibrate,shared_x,shared_y,atb_pos):
        self.active = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.var1 = c_int(0)

        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Reference_Detector", label="Reference Detector Template",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_var("VAR1",self.var1, step=1,readonly=False)
        self._bar.add_button("Start", self.start)

    def start(self):
        self.active = True

    def detect(self,img):
        """
        get called once every frame.
        reference positon need to be published to shared_x and shared_y
        if no reference was found, publish 0,0
        """
        if self.active:
            # detect a reference from the image and broadcast the result as norm. coordinates
            self.shared_x.value = 0.
            self.shared_y.value = 0.
            # make sure to set these to 0 when you are not detecting a reference
        else:
            pass

    def new_ref(self,pos):
        """
        gets called when the user clicks on the wolrd window screen
        """
        pass

    def gl_display(self):
        """Gets called and the end of each image loop.
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published

        """
        pass

    def is_done(self):
        """Gets called after detect().
        return true if the calibration routine has finished.
        the caller will then reduce the ref-count of this instance to 0 triggering __del__

        if your calibration routine does not end by itself, the use will induce the end using the gui
        in which case the ref-count will get 0 as well.

        """
        return


    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar

    def __del__(self):
        '''Do what is required for clean up. This happes when a user changes the detector. It can happen at any point

        '''
        self.global_calibrate.value = False
        self.shared_x.value = 0.
        self.shared_y.value = 0.



class Automated_White_Ring_Detector(object):
    """Detector looks for a white ring on a black background.
        Using 9 positions/points within the FOV
        Ref detector will direct one to good positions with audio cues
        Calibration only collects data at the good positions
    """
    def __init__(self,global_calibrate,shared_x,shared_y,atb_pos):
        self.active = False
        self.detected = False
        self.publish = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.x = 0
        self.y = 0
        self.r = 0
        self.w = 0
        self.counter = 0

        # sites are the nine point positions in the FOV
        self.sites = []
        self.site_size = 220 # size of the circular area



        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Automated_White_Ring_Detector", label="Automated White Ring Detector",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("  begin calibrating  ", self.start)
        self._bar.add_button("  end calibrating  ", self.stop)
        self._bar.add_separator("Sep1")
        self._bar.add_var("counter", getter=self.get_count)
        self._bar.add_var("marker detection response", getter=self.get_response)

    def start(self):
        audio.say("Starting Calibration")
        self.sites = [  (-.9,-.9), ( 0,-.9), ( .9,-.9),
                        (-.9, 0), ( 0, 0), ( .9, 0),
                        (-.9, .9), ( 0, .9), ( .9, .9)]

        self.global_calibrate.value = True
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = False

    def get_count(self):
        return self.counter

    def get_response(self):
        return self.r

    def detect(self,img):
        """
        gets called once every frame.
        reference positon need to be published to shared_x and shared_y
        if no reference was found, publish 0,0
        """
        if self.active:
            s_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # self.candidate_points = self.detector.detect(s_img)

            # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7,gray)
            # img[:] = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

            # coarse ring detection
            integral = cv2.integral(s_img)
            integral =  np.array(integral,dtype=c_float)
            row,col,w,r = ring_filter(integral)
            response_center = col+w/2.,row+w/2.
            response_threshold = 50
            # print x,y, w,r
            self.x = int(col)
            self.y = int(row)
            self.r = int(r)
            self.w = int(w)

            if r>=response_threshold:
                self.detected= True
                self.pos = normalize(response_center,(img.shape[1],img.shape[0]),flip_y=True)

                if not self.counter:
                    for i in range(len(self.sites)):
                        screen_site = denormalize(self.sites[i],(img.shape[1],img.shape[0]),flip_y=True)
                        screen_dist = np.sqrt((response_center[0]-screen_site[0])**2+(response_center[1]-screen_site[1])**2)
                        if screen_dist <= self.site_size/2.:
                            self.sites.pop(i)
                            audio.beep()
                            self.counter = 30
                            break
            else:
                self.detected= False
                self.pos = 0,0 #indicate that no reference is detected


            if self.counter and self.detected:
                self.counter -= 1
                self.shared_x.value, self.shared_y.value = self.pos
            else:
                self.shared_x.value, self.shared_y.value = 0,0

            if not self.counter and len(self.sites)==0:
                self.stop()
        else:
            pass


    def new_ref(self,pos):
        """
        gets called when the user clicks on the world window screen
        """
        pass

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.active:
            for site in self.sites:
                draw_gl_point_norm(site,size=self.site_size,color=(0.,1.,0.,.5))

        if self.active and self.detected:
            draw_gl_polyline(  [[self.x,self.y],
                                [self.x+self.w,self.y],
                                [self.x+self.w,self.y+self.w],
                                [self.x,self.y+self.w]],
                                (0.,1.,0.,.8),
                                type='Loop')
            if self.counter:
                draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))
            else:
                draw_gl_point_norm(self.pos,size=self.r,color=(1.,0.,0.,.5))
        else:
            pass

    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar

    def __del__(self):
        '''Do what is required for clean up. This happes when a user changes the detector. It can happen at any point

        '''
        self.global_calibrate.value = False
        self.shared_x.value = 0.
        self.shared_y.value = 0.


class Automated_Threshold_Ring_Detector(object):
    """Detector looks for a white ring on a black background.
        Using 9 positions/points within the FOV
        Ref detector will direct one to good positions with audio cues
        Calibration only collects data at the good positions

        Steps:
            Adaptive threshold to obtain robust edge-based image of marker
            Find contours and filter into 2 level list using RETR_CCOMP
            Fit ellipses
    """
    def __init__(self,global_calibrate,shared_x,shared_y,atb_pos):
        self.active = False
        self.detected = False
        self.publish = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.counter = 0
        # sites are the nine point positions in the FOV
        self.sites = []
        self.site_size = 100 # size of the circular area
        self.candidate_ellipses = []


        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Automated_White_Ring_Detector", label="Automated White Ring Detector",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("  begin calibrating  ", self.start)
        self._bar.add_button("  end calibrating  ", self.stop)
        self._bar.add_separator("Sep1")
        self._bar.add_var("counter", getter=self.get_count)

    def start(self):
        audio.say("Starting Calibration")
        self.sites = [  (-.9,-.9), ( 0,-.9), ( .9,-.9),
                        (-.9, 0), ( 0, 0), ( .9, 0),
                        (-.9, .9), ( 0, .9), ( .9, .9)]

        self.global_calibrate.value = True
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = False

    def get_count(self):
        return self.counter

    def detect(self,img):
        """
        gets called once every frame.
        reference positon need to be published to shared_x and shared_y
        if no reference was found, publish 0,0
        """
        if self.active:
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # self.candidate_points = self.detector.detect(s_img)

            # get threshold image used to get crisp-clean edges
            edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
            # cv2.flip(edges,1 ,dst = edges,)
            # display the image for debugging purpuses
            # img[:] = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
             # from edges to contours to ellipses CV_RETR_CCsOMP ls fr hole
            contours, hierarchy = cv2.findContours(edges,
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS


            # remove extra encapsulation
            hierarchy = hierarchy[0]
            # turn outmost list into array
            contours =  np.array(contours)
            # keep only contours                        with parents     and      children
            contained_contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]
            # turn on to debug contours
            # cv2.drawContours(img, contained_contours,-1, (0,0,255))

            # need at least 5 points to fit ellipse
            contained_contours =  [c for c in contained_contours if len(c) >= 5]

            ellipses = [cv2.fitEllipse(c) for c in contained_contours]
            self.candidate_ellipses = []
            # filter for ellipses that have similar area as the source contour
            for e,c in zip(ellipses,contained_contours):
                a,b = e[1][0]/2.,e[1][1]/2.
                if abs(cv2.contourArea(c)-np.pi*a*b) <10:
                    self.candidate_ellipses.append(e)


            def man_dist(e,other):
                return abs(e[0][0]-other[0][0])+abs(e[0][1]-other[0][1])

            def get_cluster(ellipses):
                for e in ellipses:
                    close_ones = []
                    for other in ellipses:
                        if man_dist(e,other)<10:
                            close_ones.append(other)
                    if len(close_ones)>=3:
                        # sort by major axis to return smallest ellipse first
                        close_ones.sort(key=lambda e: max(e[1]))
                        return close_ones
                return []

            self.candidate_ellipses = get_cluster(self.candidate_ellipses)



            if len(self.candidate_ellipses) > 0:
                self.detected= True
                marker_pos = self.candidate_ellipses[0][0]
                self.pos = normalize(marker_pos,(img.shape[1],img.shape[0]),flip_y=True)

                if not self.counter:
                    for i in range(len(self.sites)):
                        screen_site = denormalize(self.sites[i],(img.shape[1],img.shape[0]),flip_y=True)
                        screen_dist = np.sqrt((marker_pos[0]-screen_site[0])**2+(marker_pos[1]-screen_site[1])**2)
                        if screen_dist <= self.site_size/2.:
                            self.sites.pop(i)
                            audio.beep()
                            self.counter = 30
                            break
            else:
                self.detected = False
                self.pos = 0,0 #indicate that no reference is detected


            if self.counter and self.detected:
                self.counter -= 1
                self.shared_x.value, self.shared_y.value = self.pos
            else:
                self.shared_x.value, self.shared_y.value = 0,0

            if not self.counter and len(self.sites)==0:
                self.stop()
        else:
            pass


    def new_ref(self,pos):
        """
        gets called when the user clicks on the world window screen
        """
        pass

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.active:
            for site in self.sites:
                draw_gl_point_norm(site,size=self.site_size,color=(0.,1.,0.,.5))

        if self.active and self.detected:
            for e in self.candidate_ellipses:
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_gl_polyline(pts,(0.,1.,0,1.))

            if self.counter:
                draw_gl_point_norm(self.pos,size=self.site_size,color=(0.,1.,0.,.5))
            else:
                draw_gl_point_norm(self.pos,size=20.,color=(1.,0.,0.,.5))
        else:
            pass

    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar

    def __del__(self):
        '''Do what is required for clean up. This happes when a user changes the detector. It can happen at any point

        '''
        self.global_calibrate.value = False
        self.shared_x.value = 0.
        self.shared_y.value = 0.


class Manual_White_Ring_Detector(object):
    """Detector looks for a white ring on a black background.

    """
    def __init__(self,global_calibrate,shared_x,shared_y,atb_pos):
        self.active = False
        self.detected = False
        self.publish = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.x = 0
        self.y = 0
        self.r = 0
        self.w = 0
        self.counter = 0


        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Reference_Detector", label="Manual White Ring Detector",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("  begin calibrating  ", self.start)
        self._bar.add_button("  end calibrating  ", self.stop)
        self._bar.add_button("  sample this point", self.sample_point, key="SPACE")
        self._bar.add_separator("Sep1")
        self._bar.add_var("counter", getter=self.get_count)
        self._bar.add_var("marker detection response", getter=self.get_response)

    def start(self):
        audio.say("Starting Calibration")
        self.global_calibrate.value = True
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = False

    def get_count(self):
        return self.counter

    def get_response(self):
        return self.r

    def sample_point(self):
        audio.beep()
        self.counter = 30


    def new_ref(self,pos):
        """
        gets called when the user clicks on the world window screen
        """
        pass

    def detect(self,img):
        """
        gets called once every frame.
        reference positon need to be published to shared_x and shared_y
        if no reference was found, publish 0,0
        """
        if self.active:
            s_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # self.candidate_points = self.detector.detect(s_img)

            # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7,gray)
            # img[:] = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

            # coarse ring detection
            integral = cv2.integral(s_img)
            integral =  np.array(integral,dtype=c_float)
            row,col,w,r = ring_filter(integral)
            response_center = col+w/2.,row+w/2.
            response_threshold = 50
            # print x,y, w,r
            self.x = int(col)
            self.y = int(row)
            self.r = int(r)
            self.w = int(w)

            if r>=response_threshold:
                self.detected= True
                self.pos = normalize(response_center,(img.shape[1],img.shape[0]),flip_y=True)
            else:
                self.detected= False
                self.pos = 0,0 #indicate that no reference is detected


            if self.counter and self.detected:
                self.counter -= 1
                self.shared_x.value, self.shared_y.value = self.pos
            else:
                self.shared_x.value, self.shared_y.value = 0,0
        else:
            pass


    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """
        if self.active and self.detected:
            draw_gl_polyline(  [[self.x,self.y],
                                [self.x+self.w,self.y],
                                [self.x+self.w,self.y+self.w],
                                [self.x,self.y+self.w]],
                                (0.,1.,0.,.8),
                                type='Loop')
            if self.counter:
                draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))
            else:
                draw_gl_point_norm(self.pos,size=self.r,color=(1.,0.,0.,.5))
        else:
            pass

    def is_done(self):
        """
        gets called after detect()
        return true if the calibration routine has finished.
        the caller will then reduce the ref-count of this instance to 0 triggering __del__

        if your calibration routine does not end by itself, the use will induce the end using the gui
        in which case the ref-count will get 0 as well.
        """
        return False

    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar


    def __del__(self):
        '''Do what is required for clean up. This happes when a user changes the detector. It can happen at any point

        '''
        self.global_calibrate.value = False
        self.shared_x.value = 0.
        self.shared_y.value = 0.


class Nine_Point_Detector(object):
    """docstring for Nine_Point_"""
    def __init__(self, global_calibrate,shared_x,shared_y,shared_stage,shared_step,shared_cal9_active,shared_circle_id,auto_advance=False,atb_pos=(0,0)):
        self.active = False
        self.detected = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.var1 = c_int(0)

        self.shared_cal9_active = shared_cal9_active
        self.shared_cal9_active.value = True
        self.shared_stage = shared_stage
        self.shared_step = shared_step
        self.shared_circle_id = shared_circle_id
        self.stage = 0
        self.step = 0
        self.next = False
        self.auto_advance = auto_advance
        self.map = (0, 2, 7, 16, 21, 23, 39, 40, 42)
        self.grid_points = None
        if self.auto_advance:
            atb_lable = "Automatic 9 Point Detector"
        else:
            atb_lable = "Directed 9 Point Detector"

      # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "9_Point_Reference_Detector", label=atb_lable,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 150))
        self._bar.add_button("  begin calibrating  ", self.start)
        if not self.auto_advance:
            self._bar.add_button("  next point", self.advance, key="SPACE")
        self._bar.add_separator("Sep1")
        self._bar.add_var("9 point stage", getter=self.get_stage)
        self._bar.add_var("9 point step", getter=self.get_step)

    def get_stage(self):
        return self.stage

    def get_step(self):
        return self.step

    def start(self):
        audio.say("Starting 9 Point Calibration")
        self.global_calibrate.value = True
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.reset()
        self.publish()
        self.active = False

    def new_ref(self,pos):
        """
        gets called when the user clicks on the world window screen
        """
        pass


    def detect(self,img):
        if self.active:
            # Statemachine
            if self.step > 30:
                self.step = 0
                self.stage += 1
                self.next = False
            # done exit now (is_done() will now return True)
            if self.stage > 8:
                return
            # Detection
            self.pos = 0,0
            self.detected = False

            if self.step in range(10, 25):
                status, self.grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                if status:
                    self.detected = True
                    img_pos = self.grid_points[self.map[self.stage]][0]
                    self.pos = normalize(img_pos, (img.shape[1],img.shape[0]),flip_y=True)
            # Advance
            if self.next or self.auto_advance:
                self.step += 1

            self.publish()



    def advance(self):
        self.next=True

    def publish(self):
        self.shared_stage.value = self.stage
        self.shared_step.value = self.step
        self.shared_circle_id.value = self.map[self.stage]
        self.shared_x.value, self.shared_y.value = self.pos

    def reset(self):
        self.step = 0
        self.stage = 0
        self.is_done = False
        self.pos = 0,0

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """
        if self.detected:
            draw_gl_polyline(self.grid_points[:,0],(0.,0.,1.,.5), type="Strip")

    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar


    def __del__(self):
        self.reset()
        self.publish()
        self.global_calibrate.value = False


class Natural_Features_Detector(object):
    """Calibrate using natural features in a scene.
        Features are selected by a user by clicking on
    """
    def __init__(self,global_calibrate,shared_x,shared_y,atb_pos):
        self.first_img = None
        self.point = None
        self.count = 0
        self.detected = False
        self.active = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_x = shared_x
        self.shared_y = shared_y
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.var1 = c_int(0)
        self.r = 40.0 # radius of circle displayed

        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Reference_Detector", label="Natural Features Detector",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("Start", self.start)
        self._bar.add_button("Stop", self.stop)

    def start(self):
        audio.say("Starting Calibration")
        self.global_calibrate.value = True
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.shared_x.value = 0
        self.shared_y.value = 0
        self.active = False

    def detect(self,img):
        if self.active:
            if self.first_img is None:
                self.first_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            if self.count:
                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                nextPts, status, err = cv2.calcOpticalFlowPyrLK(self.first_img,gray,self.point,winSize=(100,100))
                if status[0]:
                    self.detected = True
                    self.point = nextPts
                    self.first_img = gray
                    nextPts = nextPts[0]
                    self.pos = normalize(nextPts,(img.shape[1],img.shape[0]),flip_y=True)
                    self.count -=1
                else:
                    self.detected = False
                    self.pos = 0,0
            else:
                self.detected = False
                self.pos = 0,0

            self.publish()

    def gl_display(self):
        if self.detected:
            draw_gl_point_norm(self.pos,size=self.r,color=(0.,1.,0.,.5))

    def publish(self):
        self.shared_x.value, self.shared_y.value = self.pos

    def new_ref(self,pos):
        self.first_img = None
        self.point = np.array([pos,],dtype=np.float32)
        self.count = 30


    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar

    def __del__(self):
        self.global_calibrate.value = False
        self.shared_x.value = 0.
        self.shared_y.value = 0.


class Camera_Intrinsics_Calibration(object):
    """Camera_Intrinsics_Calibration
        not being an actual calibration,
        this method is used to calculate camera intrinsics.

    """
    def __init__(self,global_calibrate,shared_x,shared_y, atb_pos=(0,0)):
        self.collect_new = False
        self.calculated = False
        self.obj_grid = _gen_pattern_grid((4, 11))
        self.img_points = []
        self.obj_points = []
        self.count = 10
        self.img_shape = None

        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Reference_Detector", label="Camera Calibration",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("  Capture Pattern", self.advance, key="SPACE")
        self._bar.add_var("patterns to capture", getter=self.get_count)

    def get_count(self):
        return self.count

    def advance(self):
        if self.count ==10:
            audio.say("Capture 10 calibration patterns.")
        self.collect_new = True

    def new_ref(self,pos):
        pass

    def calculate(self):
        self.calculated = True
        camera_matrix, dist_coefs = _calibrate_camera(np.asarray(self.img_points),
                                                    np.asarray(self.obj_points),
                                                    (self.img_shape[1], self.img_shape[0]))
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coefs.npy", dist_coefs)
        audio.say("Camera calibrated and saved to file")

    def detect(self,img):
        if self.collect_new:
            status, grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if status:
                self.img_points.append(grid_points)
                self.obj_points.append(self.obj_grid)
                self.collect_new = False
                self.count -=1
                if self.count in range(1,10):
                    audio.say("%i" %(self.count))
                self.img_shape = img.shape

        if not self.count and not self.calculated:
            self.calculate()

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """
        for grid_points in self.img_points:
            calib_bounds =  cv2.convexHull(grid_points)[:,0] #we dont need that extra encapsulation that opencv likes so much
            draw_gl_polyline(calib_bounds,(0.,0.,1.,.5), type="Loop")

    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar

    def __del__(self):
        pass


# shared helper functions for detectors private to the module
def _calibrate_camera(img_pts, obj_pts, img_size):
    # generate pattern size
    camera_matrix = np.zeros((3,3))
    dist_coef = np.zeros(4)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts,
                                                    img_size, camera_matrix, dist_coef)
    return camera_matrix, dist_coefs

def _gen_pattern_grid(size=(4,11)):
    pattern_grid = []
    for i in xrange(size[1]):
        for j in xrange(size[0]):
            pattern_grid.append([(2*j)+i%2,i,0])
    return np.asarray(pattern_grid, dtype='f4')







if __name__ == '__main__':

    active_detector_class = Automated_Threshold_Ring_Detector


    from glfw import *
    import atb
    from methods import normalize, denormalize, chessboard, circle_grid, gen_pattern_grid, calibrate_camera,Temp
    from uvc_capture import autoCreateCapture
    from gl_utils import adjust_gl_view, draw_gl_texture, clear_gl_screen,draw_gl_point,draw_gl_point_norm,draw_gl_polyline_norm
    from time import time

    # Callback functions
    def on_resize(w, h):
        atb.TwWindowSize(w, h);
        adjust_gl_view(w,h)

    def on_key(key, pressed):
        if not atb.TwEventKeyboardGLFW(key,pressed):
            if pressed:
                if key == GLFW_KEY_ESC:
                    on_close()

    def on_char(char, pressed):
        if not atb.TwEventCharGLFW(char,pressed):
            pass

    def on_button(button, pressed):
        if not atb.TwEventMouseButtonGLFW(button,pressed):
            if pressed:
                pos = glfwGetMousePos()
                pos = normalize(pos,glfwGetWindowSize())
                pos = denormalize(pos,(img.shape[1],img.shape[0]) ) # Position in img pixels
                ref.detector.new_ref(pos)

    def on_pos(x, y):
        if atb.TwMouseMotion(x,y):
            pass

    def on_scroll(pos):
        if not atb.TwMouseWheel(pos):
            pass

    def on_close():
        running.value=False

    running = c_bool(1)


    # Initialize capture, check if it works
    cap = autoCreateCapture(["Logitech Camera", "C525","C615","C920","C930e"],(1280,720))
    if cap is None:
        print "WORLD: Error could not create Capture"

    s, img = cap.read()
    if not s:
        print "WORLD: Error could not get image"

    height,width = img.shape[:2]

    # helpers called by the main atb bar
    def update_fps():
        old_time, bar.timestamp = bar.timestamp, time()
        dt = bar.timestamp - old_time
        if dt:
            bar.fps.value += .05 * (1 / dt - bar.fps.value)

    def set_window_size(mode,data):
        height,width = img.shape[:2]
        ratio = (1,.75,.5,.25)[mode]
        w,h = int(width*ratio),int(height*ratio)
        glfwSetWindowSize(w,h)
        data.value=mode # update the bar.value

    def get_from_data(data):
        """
        helper for atb getter and setter use
        """
        return data.value

    def advance_calibration():
        ref.detector.advance()


    # Initialize ant tweak bar - inherits from atb.Bar
    atb.init()
    bar = atb.Bar(name = "World", label="Controls",
            help="Scene controls", color=(50, 50, 50), alpha=100,valueswidth=150,
            text='light', position=(10, 10),refresh=.3, size=(300, 200))
    bar.fps = c_float(0.0)
    bar.timestamp = time()
    bar.window_size = c_int(0)
    window_size_enum = atb.enum("Display Size",{"Full":0, "Medium":1,"Half":2,"Mini":3})

    # play and record can be tied together via pointers to the objects
    # bar.play = bar.record_video
    bar.add_var("FPS", bar.fps, step=1., readonly=True)
    bar.add_var("Display_Size", vtype=window_size_enum,setter=set_window_size,getter=get_from_data,data=bar.window_size)

    # add v4l2 camera controls to a seperate ATB bar
    if cap.controls is not None:
        c_bar = atb.Bar(name="Camera_Controls", label=cap.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=(320, 10),refresh=2., size=(200, 200))

        sorted_controls = [c for c in cap.controls.itervalues()]
        sorted_controls.sort(key=lambda c: c.order)

        for control in sorted_controls:
            name = control.atb_name
            if control.type=="bool":
                c_bar.add_var(name,vtype=atb.TW_TYPE_BOOL8,getter=control.get_val,setter=control.set_val)
            elif control.type=='int':
                c_bar.add_var(name,vtype=atb.TW_TYPE_INT32,getter=control.get_val,setter=control.set_val)
                c_bar.define(definition='min='+str(control.min),   varname=name)
                c_bar.define(definition='max='+str(control.max),   varname=name)
                c_bar.define(definition='step='+str(control.step), varname=name)
            elif control.type=="menu":
                if control.menu is None:
                    vtype = None
                else:
                    vtype= atb.enum(name,control.menu)
                c_bar.add_var(name,vtype=vtype,getter=control.get_val,setter=control.set_val)
                if control.menu is None:
                    c_bar.define(definition='min='+str(control.min),   varname=name)
                    c_bar.define(definition='max='+str(control.max),   varname=name)
                    c_bar.define(definition='step='+str(control.step), varname=name)
            else:
                pass
            if control.flags == "inactive":
                pass
                # c_bar.define(definition='readonly=1',varname=control.name)

        c_bar.add_button("refresh",cap.update_from_device)
        c_bar.add_button("load defaults",cap.load_defaults)

    else:
        c_bar = None


    ref = Temp()
    g_calibrate, g_ref_x, g_ref_y = c_bool(0), c_float(0),c_float(0)
    ref.detector = active_detector_class(g_calibrate,g_ref_x,g_ref_y, (10,230))
    # Objects as variable containers

    # Initialize glfw
    glfwInit()
    height,width = img.shape[:2]
    glfwOpenWindow(width, height, 0, 0, 0, 8, 0, 0, GLFW_WINDOW)
    glfwSetWindowTitle("Ref Detector Test")
    glfwSetWindowPos(0,0)

    # Register callbacks
    glfwSetWindowSizeCallback(on_resize)
    glfwSetWindowCloseCallback(on_close)
    glfwSetKeyCallback(on_key)
    glfwSetCharCallback(on_char)
    glfwSetMouseButtonCallback(on_button)
    glfwSetMousePosCallback(on_pos)
    glfwSetMouseWheelCallback(on_scroll)

    # gl_state settings
    import OpenGL.GL as gl
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)
    del gl

    # Event loop
    while glfwGetWindowParam(GLFW_OPENED) and running.value:
        update_fps()

        # Get an image from the grabber
        s, img = cap.read()
        ref.detector.detect(img)

        # render the screen
        clear_gl_screen()
        draw_gl_texture(img)

        ref.detector.gl_display()

        atb.draw()
        glfwSwapBuffers()

    # end while running and clean-up
    print "Process closed"
    glfwCloseWindow()
    glfwTerminate()
