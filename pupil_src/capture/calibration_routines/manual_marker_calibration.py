import os
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_polyline
import calibrate

from ctypes import c_int,c_bool
import atb
import audio

from plugin import Plugin

class Manual_Marker_Calibration(Plugin):
    """Detector looks for a white ring on a black background.
        Using 9 positions/points within the FOV
        Ref detector will direct one to good positions with audio cues
        Calibration only collects data at the good positions

        Steps:
            Adaptive threshold to obtain robust edge-based image of marker
            Find contours and filter into 2 level list using RETR_CCOMP
            Fit ellipses
    """
    def __init__(self, g_pool,atb_pos=(0,0)):
        Plugin.__init__(self)
        self.active = False
        self.detected = False
        self.g_pool = g_pool
        self.pos = None
        self.smooth_pos = 0.,0.
        self.smooth_vel = 0.
        self.sample_site = (-2,-2)
        self.counter = 0
        self.counter_max = 30
        self.candidate_ellipses = []
        self.show_edges = c_bool(0)
        self.aperture = c_int(7)
        self.dist_threshold = c_int(10)
        self.area_threshold = c_int(30)
        self.world_size = None


        atb_label = "calibrate using handheld marker"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = self.__class__.__name__, label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_button("start/stop", self.start_stop, key='c')
        # self._bar.add_var("show edges",self.show_edges, group="Advanced")
        # self._bar.add_var("counter", getter=self.get_count, group="Advanced")
        # self._bar.add_var("aperture", self.aperture, min=3,step=2, group="Advanced")
        # self._bar.add_var("area threshold", self.area_threshold, group="Advanced")
        # self._bar.add_var("eccetricity threshold", self.dist_threshold, group="Advanced")

    def start_stop(self):
        if self.active:
            self.stop()
        else:
            self.start()

    def start(self):
        audio.say("Starting Calibration")
        self.active = True
        self.ref_list = []
        self.pupil_list = []


    def stop(self):
        audio.say("Stopping Calibration")
        self.smooth_pos = 0,0
        self.counter = 0
        self.active = False


        print len(self.pupil_list), len(self.ref_list)
        cal_pt_cloud = calibrate.preprocess_data(self.pupil_list,self.ref_list)
        print "Collected ", len(cal_pt_cloud), " data points."
        if len(cal_pt_cloud) < 20:
            print "Did not collect enough data."
            return
        cal_pt_cloud = np.array(cal_pt_cloud)
        self.g_pool.map_pupil = calibrate.get_map_from_cloud(cal_pt_cloud,self.world_size,verbose=True)
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud.npy'),cal_pt_cloud)



    def get_count(self):
        return self.counter

    def update(self,frame,recent_pupil_positions):
        """
        gets called once every frame.
        reference positon need to be published to shared_pos
        if no reference was found, publish 0,0
        """
        if self.active:
            img = frame.img
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

            else:
                self.detected = False
                self.pos = None #indicate that no reference is detected


            if self.detected:
                # calculate smoothed manhattan velocity
                smoother = 0.3
                smooth_pos = np.array(self.smooth_pos)
                pos = np.array(self.pos)
                new_smooth_pos = smooth_pos + smoother*(pos-smooth_pos)
                smooth_vel_vec = new_smooth_pos - smooth_pos
                smooth_pos = new_smooth_pos
                self.smooth_pos = list(smooth_pos)
                #manhattan distance for velocity
                new_vel = abs(smooth_vel_vec[0])+abs(smooth_vel_vec[1])
                self.smooth_vel = self.smooth_vel + smoother*(new_vel-self.smooth_vel)

                #distance to last sampled site
                sample_ref_dist = smooth_pos-np.array(self.sample_site)
                sample_ref_dist = abs(sample_ref_dist[0])+abs(sample_ref_dist[1])

                # start counter if ref is resting in place and not at last sample site
                if not self.counter:
                    if self.smooth_vel < 0.01 and sample_ref_dist > 0.2:
                        self.sample_site = self.smooth_pos
                        audio.beep()
                        self.counter = self.counter_max

            if self.counter and self.detected:
                self.counter -= 1
                ref = {}
                ref["norm_pos"] = self.pos
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)
                if self.counter == 0:
                    #last sample before counter done and moving on
                    audio.tink()

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['norm_pupil'] is not None:
                    self.pupil_list.append(p_pt)

            if self.world_size is None:
                self.world_size = img.shape[1],img.shape[0]

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

        if self.active:
            draw_gl_point_norm(self.smooth_pos,size=15,color=(1.,1.,0.,.5))

        if self.active and self.detected:
            for e in self.candidate_ellipses:
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_gl_polyline(pts,(0.,1.,0,1.))

            if self.counter:
                draw_gl_point_norm(self.pos,size=30.,color=(0.,1.,0.,.5))
        else:
            pass

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        self._bar.destroy()
