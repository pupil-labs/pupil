'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from gl_utils import draw_gl_polyline,adjust_gl_view,clear_gl_screen,draw_gl_point,draw_gl_point_norm,basic_gl_setup
from methods import normalize
import atb
import audio
from ctypes import c_int,c_bool,create_string_buffer

from glfw import *
from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers_simple, draw_markers,m_marker_to_screen
from reference_surface import Reference_Surface


# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Marker_Detector(Plugin):
    """docstring

    """
    def __init__(self,g_pool,atb_pos=(0,0)):
        Plugin.__init__(self)

        # all markers that are detected in the most recent frame
        self.markers = []
        # all registered surfaces
        self.surfaces = []


        #detector vars
        self.robust_detection = c_bool(1)
        self.aperture = c_int(11)
        self.min_marker_perimeter = 40

        #debug vars
        self.draw_markers = c_bool(0)

        self.window_should_open = False
        self.window_should_close = False
        self._window = None
        self.fullscreen = c_bool(0)
        self.monitor_idx = c_int(0)
        self.monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        #primary_monitor = glfwGetPrimaryMonitor()

        atb_label = "marker detection"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="marker detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))

        self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum,group="Window",)
        self._bar.add_var("fullscreen", self.fullscreen,group="Window")
        self._bar.add_button("  open Window   ", self.do_open, key='m',group="Window")
        self._bar.add_var("edge aperture",self.aperture, step=2,min=3,group="Detector")
        self._bar.add_var('robust_detection',self.robust_detection,group="Detector")
        self._bar.add_var("draw markers",self.draw_markers,group="Detector")



        atb_pos = atb_pos[0],atb_pos[1]+110
        self._bar_markers = atb.Bar(name =self.__class__.__name__+'markers', label='registered surfaces',
            help="list of registered ref surfaces", color=(50, 100, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar_markers.add_button("  add surface   ", self.add_surface, key='a')





    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def advance(self):
        pass

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = self.monitor_handles[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 1280,720

            self._window = glfwCreateWindow(height, width, "Calibration", monitor=monitor, share=glfwGetCurrentContext())
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
            glfwMakeContextCurrent(active_window)

            self.window_should_open = False


    def on_key(self,window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    self.on_close()


    def on_close(self,window=None):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False



    def add_surface(self):
        self.surfaces.append(Reference_Surface())
        self.update_bar_markers()

    def remove_surface(self,i):
        del self.surfaces[i]
        self.update_bar_markers()


    def update_bar_markers(self):
        self._bar_markers.clear()
        self._bar_markers.add_button("  add surface   ", self.add_surface, key='a')
        for s,i in zip (self.surfaces,range(len(self.surfaces))):
            self._bar_markers.add_var("%s_name"%i,s.atb_name,group=str(i),label='name')
            self._bar_markers.add_button("%s_remove"%i, self.remove_surface,data=i,label='remove',group=str(i))
            self._bar_markers.add_var("%s detected registered"%i,create_string_buffer(512), getter=s.atb_marker_status,group=str(i),label='registered_markers' )


    def update(self,frame,recent_pupil_positions):
        img = frame.img
        if self.robust_detection.value:
            self.markers = detect_markers_robust(img,grid_size = 5,prev_markers=self.markers,min_marker_perimeter=self.min_marker_perimeter,aperture=self.aperture.value,visualize=0)
        else:
            self.markers = detect_markers_simple(img,grid_size = 5,min_marker_perimeter=self.min_marker_perimeter,aperture=self.aperture.value,visualize=0)

        if self.draw_markers.value:
            draw_markers(img,self.markers)

        for s in self.surfaces:
            s.locate(self.markers)

        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

    def gl_display(self):
        """
        for debugging now
        """

        for m in self.markers:
            if m['id'] !=-1:
                hat = np.array([[[0,0],[0,1],[.5,1.5],[1,1],[1,0],[0,0]]],dtype=np.float32)
                # hat = np.array([[[-2,-2],[-2,3],[-2,3.5],[3,3],[3,-2],[-2,-2]]],dtype=np.float32)
                hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
                draw_gl_polyline(hat.reshape((6,2)),(0.1,1.,1.,.5))

        for s in  self.surfaces:
            s.gl_draw()

        if self._window:
            self.gl_display_in_window()

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)



    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self._window:
            self.close_window()
        self._bar.destroy()
        self._bar_markers.destroy()

