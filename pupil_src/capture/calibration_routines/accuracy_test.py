'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''



import os
import cv2
import numpy as np
import scipy.spatial as sp

from methods import normalize,denormalize
from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_points_norm,draw_gl_polyline,draw_gl_polyline_norm,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from OpenGL.GLU import gluOrtho2D
import calibrate
from circle_detector import get_canditate_ellipses

from ctypes import c_int,c_bool,c_float
import atb
import audio

from plugin import Plugin

#logging
import logging
logger = logging.getLogger(__name__)


def draw_circle(pos,r,c):
    pts = cv2.ellipse2Poly(tuple(pos),(r,r),0,0,360,10)
    draw_gl_polyline(pts,c,'Polygon')

def draw_marker(pos):
    pos = int(pos[0]),int(pos[1])
    black = (0.,0.,0.,1.)
    white = (1.,1.,1.,1.)
    for r,c in zip((50,40,30,20,10),(black,white,black,white,black)):
        draw_circle(pos,r,c)


# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)


class Accuracy_Test(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between

    """
    def __init__(self, g_pool, atb_pos=(0,0)):
        Plugin.__init__(self)
        self.g_pool = g_pool
        self.active = False
        self.detected = False
        self.screen_marker_state = 0
        self.screen_marker_max = 70 # maximum bound for state
        self.active_site = 0
        self.sites = []
        self.display_pos = None
        self.on_position = False

        self.candidate_ellipses = []
        self.pos = None

        #result calculation variables:
        self.fow = c_float(90.) #taken from c930e specsheet, confirmed though mesurement within ~10deg.
        self.res =  c_float(np.sqrt(1280**2 + 720**2))
        self.outlier_thresh = c_float(5.)
        self.accuray = c_float(0)
        self.percision = c_float(0)

        try:
            self.pt_cloud = np.load(os.path.join(self.g_pool.user_dir,'accuray_test_pt_cloud.npy'))
            gaze,ref = self.pt_cloud[:,0:2],self.pt_cloud[:,2:4]
            error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
            self.error_lines = error_lines.reshape(-1,2)
        except Exception:
            self.error_lines = None
            self.pt_cloud = None


        self.show_edges = c_bool(0)
        self.dist_threshold = c_int(5)
        self.area_threshold = c_int(20)

        self.world_size = None

        self._window = None
        self.window_should_close = False
        self.window_should_open = False
        self.fullscreen = c_bool(1)
        self.monitor_idx = c_int(0)
        self.monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        #primary_monitor = glfwGetPrimaryMonitor()



        atb_label = "screen marker based accuracy test"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = self.__class__.__name__, label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum)
        self._bar.add_var("fullscreen", self.fullscreen)
        self._bar.add_button("  start test  ", self.start, key='c')

        self._bar.add_var('diagonal FOV',self.fow)
        self._bar.add_var('diagonal resolution',self.res,readonly= True)
        self._bar.add_var('outlier threshold deg',self.outlier_thresh)
        self._bar.add_var('angular accuray',self.accuray,readonly=True)
        self._bar.add_var('angular percision',self.percision,readonly=True)
        self._bar.add_button('calculate result',self.calc_result)
        self._bar.add_separator("Sep1")
        self._bar.add_var("show edges",self.show_edges)
        self._bar.add_var("area threshold", self.area_threshold)
        self._bar.add_var("eccetricity threshold", self.dist_threshold)


    def start(self):
        if self.active:
            return

        audio.say("Starting Accuracy_Test")
        logger.info("Starting Accuracy_Test")
        self.sites = [  (.5, .5), (0,.5),
                        (0.,1),(.5,1),(1.,1.),
                        (1,.5),
                        (1., 0),(.5, 0),(0,0.),
                        (.5,.5),(.5,.5)]
        self.sites = np.random.random((10,2)).tolist() + self.sites
        self.active_site = 0
        self.active = True
        self.ref_list = []
        self.gaze_list = []
        self.window_should_open = True

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = self.monitor_handles[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 640,360

            self._window = glfwCreateWindow(height, width, "Accuracy_Test", monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen.value:
                glfwSetWindowPos(self._window,200,0)

            on_resize(self._window,height,width)

            #Register callbacks
            glfwSetWindowSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)
            self.window_should_open = False


    def on_key(self,window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    self.stop()

    def on_close(self,window=None):
        if self.active:
            self.stop()

    def stop(self):
        audio.say("Stopping Accuracy_Test")
        logger.info('Stopping Accuracy_Test')
        self.screen_marker_state = 0
        self.active = False
        self.window_should_close = True

        pt_cloud = preprocess_data_gaze(self.gaze_list,self.ref_list)

        logger.info("Collected %s data points." %len(pt_cloud))

        if len(pt_cloud) < 20:
            logger.warning("Did not collect enough data.")
            return

        pt_cloud = np.array(pt_cloud)
        np.save(os.path.join(self.g_pool.user_dir,'accuray_test_pt_cloud.npy'),pt_cloud)
        gaze,ref = pt_cloud[:,0:2],pt_cloud[:,2:4]
        error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
        self.error_lines = error_lines.reshape(-1,2)
        self.pt_cloud = pt_cloud


    def calc_result(self):
        #lets denormalize:
        # test world cam resolution
        if self.pt_cloud == None:
            logger.warning("Please run test first!")
            return

        if self.world_size == None:
            return

        pt_cloud = self.pt_cloud.copy()
        res = self.world_size
        pt_cloud[:,0:3:2] *= res[0]
        pt_cloud[:,1:4:2] *= res[1]

        field_of_view = self.fow.value
        px_per_degree = self.res.value/field_of_view

        # Accuracy is calculated as the average angular
        # offset (distance) (in degrees of visual angle)
        # between fixations locations and the corresponding
        # locations of the fixation targets.

        gaze,ref = pt_cloud[:,0:2],pt_cloud[:,2:4]
        # site = pt_cloud[:,4]
        error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
        error_lines = error_lines.reshape(-1,2)
        error_mag = sp.distance.cdist(gaze,ref).diagonal().copy()
        accuray_pix = np.mean(error_mag)
        logger.info("Gaze error mean in world camera pixel: %f"%accuray_pix)
        error_mag /= px_per_degree
        logger.info('Error in degrees: %s'%error_mag)
        logger.info('Outliers: %s'%np.where(error_mag>=self.outlier_thresh.value))
        self.accuray.value = np.mean(error_mag[error_mag<self.outlier_thresh.value])
        logger.info('Angular accuray: %s'%self.accuray.value)


        #lets calculate percision:  (RMS of distance of succesive samples.)
        # This is a little rough as we do not compensate headmovements in this test.

        # Precision is calculated as the Root Mean Square (RMS)
        # of the angular distance (in degrees of visual angle)
        # between successive samples during a fixation
        succesive_distances_gaze = sp.distance.cdist(gaze[:-1],gaze[1:]).diagonal().copy()
        succesive_distances_ref = sp.distance.cdist(ref[:-1],ref[1:]).diagonal().copy()
        succesive_distances_gaze /=px_per_degree
        succesive_distances_ref /=px_per_degree
        # if the ref distance is to big we must have moved to a new fixation or there is headmovement,
        # if the gaze dis is to big we can assume human error
        # both times gaze data is not valid for this mesurement
        succesive_distances =  succesive_distances_gaze[np.logical_and(succesive_distances_gaze< 1., succesive_distances_ref< .1)]
        self.percision.value = np.sqrt(np.mean(succesive_distances**2))
        logger.info("Angular percision: %s"%self.percision.value)

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False


    def update(self,frame,recent_pupil_positions,events):

        #get world image size for error fitting later.
        if self.world_size is None:
            self.world_size = frame.img.shape[1],frame.img.shape[0]

        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

        if self.active:
            img = frame.img

            #detect the marker
            self.candidate_ellipses = get_canditate_ellipses(img,
                                                            area_threshold=self.area_threshold.value,
                                                            dist_threshold=self.dist_threshold.value,
                                                            min_ring_count=4,
                                                            visual_debug=self.show_edges.value)

            if len(self.candidate_ellipses) > 0:
                self.detected= True
                marker_pos = self.candidate_ellipses[0][0]
                self.pos = normalize(marker_pos,(img.shape[1],img.shape[0]),flip_y=True)

            else:
                self.detected = False
                self.pos = None #indicate that no reference is detected


            #only save a valid ref position if within sample window of calibraiton routine
            on_position = 0 < self.screen_marker_state < self.screen_marker_max-50
            if on_position and self.detected:
                ref = {}
                ref["norm_pos"] = self.pos
                ref["timestamp"] = frame.timestamp
                ref['site'] = self.active_site
                self.ref_list.append(ref)

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['norm_gaze'] is not None:
                    self.gaze_list.append(p_pt)

            # Animate the screen marker
            if self.screen_marker_state < self.screen_marker_max:
                if self.detected or not on_position:
                    self.screen_marker_state += 1
            else:
                self.screen_marker_state = 0
                self.active_site += 1
                logger.debug("Moving screen marker to site no %s"%self.active_site)
                if self.active_site == len(self.sites)-2:
                    self.stop()
                    return

            # function to smoothly interpolate between points input:(0-screen_marker_max) output: (0-1)
            m, s = self.screen_marker_max, self.screen_marker_state

            interpolation_weight = np.tanh(((s-2/3.*m)*4.)/(1/3.*m))*(-.5)+.5

            #use np.arrays for per element wise math
            current = np.array(self.sites[self.active_site])
            next = np.array(self.sites[self.active_site+1])
            # weighted sum to interpolate between current and next
            new_pos =  current * interpolation_weight + next * (1-interpolation_weight)
            #broadcast next commanded marker postion of screen
            self.display_pos = list(new_pos)
            self.on_position = on_position




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
        if self._window:
            self.gl_display_in_window()


        if not self.active and self.error_lines is not None:
            draw_gl_polyline_norm(self.error_lines,(1.,0.5,0.,.5),type='Lines')
            draw_gl_points_norm(self.error_lines[1::2],color=(.0,0.5,0.5,.5),size=3)
            draw_gl_points_norm(self.error_lines[0::2],color=(.5,0.0,0.0,.5),size=3)



    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        # Set Matrix unsing gluOrtho2D to include padding for the marker of radius r
        #
        ############################
        #            r             #
        # 0,0##################w,h #
        # #                      # #
        # #                      # #
        #r#                      #r#
        # #                      # #
        # #                      # #
        # 0,h##################w,h #
        #            r             #
        ############################
        r = 60
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetWindowSize(self._window)
        # compensate for radius of marker
        gluOrtho2D(-r,p_window_size[0]+r,p_window_size[1]+r, -r) # origin in the top left corner just like the img np-array
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        screen_pos = denormalize(self.display_pos,p_window_size,flip_y=True)

        draw_marker(screen_pos)
        #some feedback on the detection state

        if self.detected and self.on_position:
            draw_gl_point(screen_pos, 5, (0.,1.,0.,1.))
        else:
            draw_gl_point(screen_pos, 5, (1.,0.,0.,1.))

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        if self.active:
            self.stop()
        if self._window:
            self.close_window()
        self._bar.destroy()


def preprocess_data_gaze(gaze_pts,ref_pts):
    '''small utility function to deal with timestamped but uncorrelated data
    input must be lists that contain dicts with at least "timestamp",'norm_pos' and "norm_gaze""
    '''
    correlated_data = []

    if len(ref_pts)<=2:
        return correlated_data

    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = []
        while gaze_pts:
            #select all points past the half-way point between current and next ref data sample
            if gaze_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                matched.append(gaze_pts.pop(0))
            else:
                for p_pt in matched:
                    #only use close points
                    if abs(p_pt['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                        data_pt = p_pt["norm_gaze"][0], p_pt["norm_gaze"][1],cur_ref_pt['norm_pos'][0],cur_ref_pt['norm_pos'][1],cur_ref_pt['site']
                        correlated_data.append(data_pt)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break
    return correlated_data
