'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,adjust_gl_view,draw_gl_point_norm,draw_gl_polyline,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from OpenGL.GLU import gluOrtho2D
import calibrate
from circle_detector import get_candidate_ellipses

import audio

from pyglui import ui
from pyglui.cygl.utils import init as cygl_init
from pyglui.cygl.utils import draw_points as cygl_draw_points
from pyglui.cygl.utils import RGBA as cygl_rgba


from plugin import Calibration_Plugin
from gaze_mappers import Simple_Gaze_Mapper

#logging
import logging
logger = logging.getLogger(__name__)


def draw_marker(pos):
    pos = int(pos[0]),int(pos[1])
    black = cygl_rgba(0.,0.,0.,1.)
    white = cygl_rgba(1.,1.,1.,1.)
    for r,c in zip((50,40,30,20,10),(black,white,black,white,black)):
        cygl_draw_points([pos],size=r,color=c,sharpness=0.9)

# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    hdpi_factor = glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0]
    w,h = w*hdpi_factor, h*hdpi_factor
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)


class Screen_Marker_Calibration(Calibration_Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between

    """
    def __init__(self, g_pool,menu_conf = {'collapsed':True},fullscreen = True):
        super(Screen_Marker_Calibration, self).__init__(g_pool)
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

        self.show_edges = 0
        self.dist_threshold = 5
        self.area_threshold = 20

        self.world_size = None

        self._window = None

        self.menu = None
        self.menu_conf = menu_conf
        self.button = None

        self.fullscreen = fullscreen


    def init_gui(self):
        self.monitor_idx = 0
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]

        #primary_monitor = glfwGetPrimaryMonitor()
        self.info = ui.Info_Text("Calibrate gaze parameters using a screen based animation.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.menu.configuration = self.menu_conf
        self.g_pool.calibration_menu.append(self.menu)
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(self.monitor_names)),labels=self.monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use Fullscreen'))

        submenu = ui.Growing_Menu('Advanced')
        submenu.collapsed = True
        self.menu.append(submenu)
        submenu.append(ui.Switch('show_edges',self,label='show edges'))
        submenu.append(ui.Slider('area_threshold',self,step=1,min=5,max=50,label='Area Threshold'))
        submenu.append(ui.Slider('dist_threshold',self,step=.5,min=1,max=20,label='Eccetricity Threshold'))

        self.button = ui.Thumb('active',self,setter=self.toggle,label='Calibrate',hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.append(self.button)


    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.calibration_menu.remove(self.menu)
            self.g_pool.calibration_menu.remove(self.info)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def toggle(self,new_var):
        if self.active:
            self.stop()
        else:
            self.start()



    def start(self):
        audio.say("Starting Calibration")
        logger.info("Starting Calibration")
        self.sites = [  (.25, .5),(.25, .5), (0,.5),
                        (0.,1),(.5,1),(1.,1.),
                        (1,.5),
                        (1., 0),(.5, 0),(0,0.),
                        (.75,.5)]

        self.active_site = 0
        self.active = True
        self.ref_list = []
        self.pupil_list = []
        self.open_window()

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 640,360

            self._window = glfwCreateWindow(height, width, "Calibration", monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
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
            
            # initalize cygl
            cygl_init()
            

    def on_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.stop()

    def on_close(self,window=None):
        if self.active:
            self.stop()

    def stop(self):
        audio.say("Stopping Calibration")
        logger.info('Stopping Calibration')
        self.screen_marker_state = 0
        self.active = False
        self.close_window()
        self.button.status_text = ''

        cal_pt_cloud = calibrate.preprocess_data(self.pupil_list,self.ref_list)

        logger.info("Collected %s data points." %len(cal_pt_cloud))

        if len(cal_pt_cloud) < 20:
            logger.warning("Did not collect enough data.")
            return

        cal_pt_cloud = np.array(cal_pt_cloud)
        map_fn = calibrate.get_map_from_cloud(cal_pt_cloud,self.world_size)
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud.npy'),cal_pt_cloud)

        # prepare destruction of current gaze_mapper... and remove it
        for p in self.g_pool.plugins:
            if p.base_class_name == 'Gaze_Mapping_Plugin':
                p.alive = False
        self.g_pool.plugins = [p for p in self.g_pool.plugins if p.alive]

        #add new gaze mapper
        self.g_pool.plugins.append(Simple_Gaze_Mapper(self.g_pool,map_fn))
        self.g_pool.plugins.sort(key=lambda p: p.order)



    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None


    def update(self,frame,events):
        if self.active:
            recent_pupil_positions = events['pupil_positions']
            gray_img = frame.gray

            #get world image size for error fitting later.
            if self.world_size is None:
                self.world_size = frame.width,frame.height

            #detect the marker
            self.candidate_ellipses = get_candidate_ellipses(gray_img,
                                                            area_threshold=self.area_threshold,
                                                            dist_threshold=self.dist_threshold,
                                                            min_ring_count=4,
                                                            visual_debug=self.show_edges)

            if len(self.candidate_ellipses) > 0:
                self.detected= True
                marker_pos = self.candidate_ellipses[0][0]
                self.pos = normalize(marker_pos,(frame.width,frame.height),flip_y=True)

            else:
                self.detected = False
                self.pos = None #indicate that no reference is detected


            #only save a valid ref position if within sample window of calibraiton routine
            on_position = 40 < self.screen_marker_state < self.screen_marker_max-5
            if on_position and self.detected:
                ref = {}
                ref["norm_pos"] = self.pos
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['confidence'] > self.g_pool.pupil_confidence_threshold:
                    self.pupil_list.append(p_pt)

            # Animate the screen marker
            if self.screen_marker_state < self.screen_marker_max:
                if self.detected or not on_position:
                    self.screen_marker_state += 1
            else:
                self.screen_marker_state = 0
                self.active_site += 1
                logger.debug("Moving screen marker to site no %s"%self.active_site)
                if self.active_site == 10:
                    self.stop()
                    return

            # function to smoothly interpolate between points input:(0-screen_marker_max) output: (0-1)
            m, s = self.screen_marker_max, self.screen_marker_state

            interpolation_weight = np.tanh(((s-1/6.*m)*10.)/(5/6.*m))*(-.5)+.5

            #use np.arrays for per element wise math
            current = np.array(self.sites[self.active_site])
            next = np.array(self.sites[self.active_site+1])
            # weighted sum to interpolate between current and next
            new_pos =  current * interpolation_weight + next * (1-interpolation_weight)
            #broadcast next commanded marker postion of screen
            self.display_pos = list(new_pos)
            self.on_position = on_position
            self.button.status_text = '%s / %s'%(self.active_site,9)




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
                # draw_gl_polyline(pts,(0.,1.,0,1.))
        else:
            pass
        if self._window:
            self.gl_display_in_window()


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
            cygl_draw_points([screen_pos],size=5,color=cygl_rgba(0.,1.,0.,1.),sharpness=0.95)
        else:
            cygl_draw_points([screen_pos],size=5,color=cygl_rgba(1.,0.,0.,1.),sharpness=0.95)

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


    def get_init_dict(self):
        d = {}
        d['fullscreen'] = self.fullscreen
        if self.menu:
            d['menu_conf'] = self.menu.configuration
        else:
            d['menu_conf'] = self.menu_conf
        return d

    def cleanup(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.active:
            self.stop()
        if self._window:
            self.close_window()
        self.deinit_gui()
