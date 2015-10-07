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
from file_methods import save_object
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup
from methods import normalize


import OpenGL.GL as gl
from pyglui import ui
from pyglui.cygl.utils import draw_polyline,draw_points,RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from glfw import *

from plugin import Calibration_Plugin

#logging
import logging
logger = logging.getLogger(__name__)


# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Camera_Intrinsics_Estimation(Calibration_Plugin):
    """Camera_Intrinsics_Calibration
        This method is not a gaze calibration.
        This method is used to calculate camera intrinsics.

    """
    def __init__(self,g_pool,fullscreen = False):
        super(Camera_Intrinsics_Estimation, self).__init__(g_pool)
        self.collect_new = False
        self.calculated = False
        self.obj_grid = _gen_pattern_grid((4, 11))
        self.img_points = []
        self.obj_points = []
        self.count = 10
        self.img_shape = None

        self.display_grid = _make_grid()

        self._window = None

        self.menu = None
        self.button = None
        self.clicks_to_close = 5
        self.window_should_close = False
        self.fullscreen = fullscreen
        self.monitor_idx = 0


        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center')



    def init_gui(self):

        monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]
        #primary_monitor = glfwGetPrimaryMonitor()
        self.info = ui.Info_Text("Estimate Camera intrinsics of the world camera. This is only used for 3D marker tracking at the moment. Using an 11x9 asymmetrical circle grid. Click 'C' to capture a pattern.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.menu.append(ui.Button('show Pattern',self.open_window))
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(monitor_names)),labels=monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use Fullscreen'))
        self.g_pool.calibration_menu.append(self.menu)

        self.button = ui.Thumb('collect_new',self,setter=self.advance,label='Capture',hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.calibration_menu.remove(self.menu)
            self.g_pool.calibration_menu.remove(self.info)

            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None

    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def get_count(self):
        return self.count

    def advance(self,_):
        if self.count ==10:
            logger.info("Capture 10 calibration patterns.")
            self.button.status_text = "%i to go" %(self.count)

        self.collect_new = True

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width = 640,480

            self._window = glfwCreateWindow(height, width, "Calibration", monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window,200,0)


            #Register callbacks
            glfwSetFramebufferSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)
            glfwSetMouseButtonCallback(self._window,self.on_button)

            on_resize(self._window,*glfwGetFramebufferSize(self._window))


            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            glfwMakeContextCurrent(active_window)

            self.clicks_to_close = 5



    def on_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.on_close()


    def on_button(self,window,button, action, mods):
        if action ==GLFW_PRESS:
            self.clicks_to_close -=1
        if self.clicks_to_close ==0:
            self.on_close()


    def on_close(self,window=None):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None


    def calculate(self):
        self.calculated = True
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(np.array(self.obj_points), np.array(self.img_points),self.g_pool.capture.frame_size)
        logger.info("Calibrated Camera, RMS:%s"%rms)
        camera_calibration = {'camera_matrix':camera_matrix,'dist_coefs':dist_coefs,'camera_name':self.g_pool.capture.name,'resolution':self.g_pool.capture.frame_size}
        save_object(camera_calibration,os.path.join(self.g_pool.user_dir,"camera_calibration"))
        logger.info("Calibration saved to user folder")

    def update(self,frame,events):
        if self.collect_new:
            img = frame.img
            status, grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if status:
                self.img_points.append(grid_points)
                self.obj_points.append(self.obj_grid)
                self.collect_new = False
                self.count -=1
                self.img_shape = img.shape
                self.button.status_text = "%i to go"%(self.count)


        if self.count<=0 and not self.calculated:
            self.calculate()
            self.button.status_text = ''

        if self.window_should_close:
            self.close_window()


    def gl_display(self):

        for grid_points in self.img_points:
            calib_bounds =  cv2.convexHull(grid_points)[:,0] #we dont need that extra encapsulation that opencv likes so much
            draw_polyline(calib_bounds,1,RGBA(0.,0.,1.,.5),line_type=gl.GL_LINE_LOOP)

        if self._window:
            self.gl_display_in_window()

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetWindowSize(self._window)
        r = p_window_size[0]/15.
        # compensate for radius of marker
        gl.glOrtho(-r,p_window_size[0]+r,p_window_size[1]+r,-r ,-1,1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        #hacky way of scaling and fitting in different window rations/sizes
        grid = _make_grid()*min((p_window_size[0],p_window_size[1]*5.5/4.))
        #center the pattern
        grid -= np.mean(grid)
        grid +=(p_window_size[0]/2-r,p_window_size[1]/2+r)

        draw_points(grid,size=r,color=RGBA(0.,0.,0.,1),sharpness=0.95)

        if self.clicks_to_close <5:
            self.glfont.set_size(int(p_window_size[0]/30.))
            self.glfont.draw_text(p_window_size[0]/2.,p_window_size[1]/4.,'Touch %s more times to close window.'%self.clicks_to_close)

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


    def get_init_dict(self):
        return {}


    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have a gui or glfw window destroy it here.
        """
        if self._window:
            self.close_window()
        self.deinit_gui()


def _gen_pattern_grid(size=(4,11)):
    pattern_grid = []
    for i in xrange(size[1]):
        for j in xrange(size[0]):
            pattern_grid.append([(2*j)+i%2,i,0])
    return np.asarray(pattern_grid, dtype='f4')


def _make_grid(dim=(11,4)):
    """
    this function generates the structure for an asymmetrical circle grid
    domain (0-1)
    """
    x,y = range(dim[0]),range(dim[1])
    p = np.array([[[s,i] for s in x] for i in y], dtype=np.float32)
    p[:,1::2,1] += 0.5
    p = np.reshape(p, (-1,2), 'F')

    # scale height = 1
    x_scale =  1./(np.amax(p[:,0])-np.amin(p[:,0]))
    y_scale =  1./(np.amax(p[:,1])-np.amin(p[:,1]))

    p *=x_scale,x_scale/.5

    return p


