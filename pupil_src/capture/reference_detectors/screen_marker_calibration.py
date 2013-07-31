import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_polyline

from ctypes import c_int,c_bool
import atb
import audio

from plugin import Plugin

class Screen_Marker_Calibration(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between

    """
    def __init__(self, global_calibrate, shared_pos, screen_marker_pos, screen_marker_state, atb_pos=(0,0)):
        Plugin.__init__(self)

        self.active = False
        self.detected = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False

        self.shared_pos = shared_pos

        self.shared_screen_marker_pos = screen_marker_pos
        self.shared_screen_marker_state = screen_marker_state # used for v
        self.screen_marker_state = 0
        self.screen_marker_max = 90 # maximum bound for state
        self.pos = 0,0 # 0,0 is used to indicate no point detected


        self.active_site = 0
        self.sites = []


        self.candidate_ellipses = []

        self.show_edges = c_bool(0)
        self.aperture = c_int(7)
        self.dist_threshold = c_int(10)
        self.area_threshold = c_int(30)


        atb_label = "calibrate on screen"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = self.__class__.__name__, label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 80))
        self._bar.add_button("  start calibrating  ", self.start, key='c')
        self._bar.add_separator("Sep1")
        self._bar.add_var("show edges",self.show_edges)
        self._bar.add_var("aperture", self.aperture, min=3,step=2)
        self._bar.add_var("area threshold", self.area_threshold)
        self._bar.add_var("eccetricity threshold", self.dist_threshold)


    def start(self):
        audio.say("Starting Calibration")

        c = 1.
        self.sites = [  (.0, 0),
                        (-c,c), (0.,c),(c,c),
                        (c,0.),
                        (c,-c), (0., -c),( -c, -c),
                        (-c,0.),
                        (.0,0.),(.0,0.)]

        self.active_site = 0
        self.shared_screen_marker_state.value = 1
        self.global_calibrate.value = True
        self.shared_pos[:] = 0,0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.screen_marker_state = 0
        self.shared_screen_marker_state.value = 0
        self.global_calibrate.value = False
        self.reset()
        self.publish()
        self.active = False

    def new_ref(self,pos):
        """
        gets called when the user clicks on the world window screen
        """
        pass


    def advance(self):
        self.next=True

    def publish(self):
        self.shared_pos[:] = self.pos

    def reset(self):
        self.pos = 0,0

    def update(self,img):
        if self.active:
            #detect the marker
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # self.candidate_points = self.detector.detect(s_img)

            # get threshold image used to get crisp-clean edges
            edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.aperture.value, 7)
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
                # retrun the first cluser of at least 3 concetric ellipses
                for e in ellipses:
                    close_ones = []
                    for other in ellipses:
                        if man_dist(e,other)<self.dist_threshold.value:
                            close_ones.append(other)
                    if len(close_ones)>=4:
                        # sort by major axis to return smallest ellipse first
                        close_ones.sort(key=lambda e: max(e[1]))
                        return close_ones
                return []

            self.candidate_ellipses = get_cluster(self.candidate_ellipses)



            if len(self.candidate_ellipses) > 0:
                self.detected= True
                marker_pos = self.candidate_ellipses[0][0]
                self.pos = normalize(marker_pos,(img.shape[1],img.shape[0]),flip_y=True)

            else:
                self.detected = False
                self.pos = 0,0 #indicate that no reference is detected


            #only broadcast a valid ref position if within sample window of calibraiton routine
            if 0< self.screen_marker_state < self.screen_marker_max-50:
                pass
            else:
                self.pos = 0,0

            self.publish()
            # Animate the screen marker
            if self.screen_marker_state < self.screen_marker_max:
                if self.detected:
                    self.screen_marker_state += 1
            else:
                self.screen_marker_state = 0
                self.active_site += 1
                print self.active_site
                if self.active_site == 10:
                    self.stop()
                    return

            # function to smoothly interpolate between points input:(0-90) output: (0-1)
            interpolation_weight = np.tanh(((self.screen_marker_state-2/3.*self.screen_marker_max)*4.)/(1/3.*self.screen_marker_max))*(-.5)+.5

            #use np.arrays for per element wise math
            current = np.array(self.sites[self.active_site])
            next = np.array(self.sites[self.active_site+1])
            # weighted sum to interpolate between current and next
            new_pos =  current * interpolation_weight + next * (1-interpolation_weight)
            #broadcast next commanded marker postion of screen
            self.shared_screen_marker_pos[:] = list(new_pos)


    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.active and self.detected:
            for e in self.candidate_ellipses:
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_gl_polyline(pts,(0.,1.,0,1.))
        else:
            pass


    def __del__(self):
        self.reset()
        self.publish()
        self.global_calibrate.value = False