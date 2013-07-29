
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_polyline

import atb
import audio

class Automated_Threshold_Ring_Detector(Plugin):
    """Detector looks for a white ring on a black background.
        Using 9 positions/points within the FOV
        Ref detector will direct one to good positions with audio cues
        Calibration only collects data at the good positions

        Steps:
            Adaptive threshold to obtain robust edge-based image of marker
            Find contours and filter into 2 level list using RETR_CCOMP
            Fit ellipses
    """
    def __init__(self, global_calibrate,shared_pos,screen_marker_pos,screen_marker_state,atb_pos=(0,0)):
        Plugin.__init__()
        self.active = False
        self.detected = False
        self.publish = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_pos = shared_pos
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.counter = 0
        # sites are the nine point positions in the FOV
        self.sites = []
        self.site_size = 100 # size of the circular area
        self.candidate_ellipses = []

        self.show_edges = c_bool(1)
        self.apature = c_int(7)
        self.dist_threshold = c_int(10)
        self.area_threshold = c_int(30)

        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "Automated_White_Ring_Detector", label="Automated White Ring Detector",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("  begin calibrating  ", self.start)
        self._bar.add_button("  end calibrating  ", self.stop)
        self._bar.add_separator("Sep1")
        self._bar.add_var("show edges",self.show_edges)
        self._bar.add_var("counter", getter=self.get_count)
        self._bar.add_var("apature", self.apature, min=3,step=2)
        self._bar.add_var("area threshold", self.area_threshold)
        self._bar.add_var("eccetricity threshold", self.dist_threshold)

    def start(self):
        audio.say("Starting Calibration")
        self.sites = [  (-.9,-.9), ( 0,-.9), ( .9,-.9),
                        (-.9, 0), ( 0, 0), ( .9, 0),
                        (-.9, .9), ( 0, .9), ( .9, .9)]

        self.global_calibrate.value = True
        self.shared_pos[:] = 0,0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.shared_pos[:] = 0,0
        self.active = False

    def get_count(self):
        return self.counter

    def update(self,img):
        """
        gets called once every frame.
        reference positon need to be published to shared_pos
        if no reference was found, publish 0,0
        """
        if self.active:
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # self.candidate_points = self.detector.detect(s_img)

            # get threshold image used to get crisp-clean edges
            edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.apature.value, 7)
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
            if self.show_edges.value:
                cv2.drawContours(img, contained_contours,-1, (0,0,255))

            # need at least 5 points to fit ellipse
            contained_contours =  [c for c in contained_contours if len(c) >= 5]

            ellipses = [cv2.fitEllipse(c) for c in contained_contours]
            self.candidate_ellipses = []
            # filter for ellipses that have similar area as the source contour
            for e,c in zip(ellipses,contained_contours):
                a,b = e[1][0]/2.,e[1][1]/2.
                if abs(cv2.contourArea(c)-np.pi*a*b) <self.area_threshold.value:
                    self.candidate_ellipses.append(e)


            def man_dist(e,other):
                return abs(e[0][0]-other[0][0])+abs(e[0][1]-other[0][1])

            def get_cluster(ellipses):
                for e in ellipses:
                    close_ones = []
                    for other in ellipses:
                        if man_dist(e,other)<self.dist_threshold.value:
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
                self.shared_pos[:] = self.pos
            else:
                self.shared_pos[:] = 0,0

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

    def __del__(self):
        '''Do what is required for clean up. This happes when a user changes the detector. It can happen at any point

        '''
        self.global_calibrate.value = False
        self.shared_pos[:] = 0.,0.
