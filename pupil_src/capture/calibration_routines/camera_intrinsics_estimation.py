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
from gl_utils import draw_gl_polyline,adjust_gl_view,clear_gl_screen,draw_gl_point,draw_gl_point_norm,basic_gl_setup
from methods import normalize
import audio


from pyglui import ui
from plugin import Calibration_Plugin


from glfw import *
#logging
import logging
logger = logging.getLogger(__name__)


# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    hdpi_factor = glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0]
    w,h = w*hdpi_factor, h*hdpi_factor
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Camera_Intrinsics_Estimation(Calibration_Plugin):
    """Camera_Intrinsics_Calibration
        not being an actual calibration,
        this method is used to calculate camera intrinsics.

    """
    def __init__(self,g_pool):
        super(Camera_Intrinsics_Estimation, self).__init__(g_pool)
        self.collect_new = False
        self.g_pool = g_pool
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


    def init_gui(self):
        self.fullscreen = True
        self.monitor_idx = 0
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]

        #primary_monitor = glfwGetPrimaryMonitor()

        self.menu = ui.Growing_Menu('Screen Based Calibration')
        self.g_pool.sidebar.append(self.menu)

        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(self.monitor_names)),labels=self.monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use Fullscreen'))
        self.menu.append(ui.Button('show Pattern',self.open_window))

        self.button = ui.Thumb('collect_new',self,setter=self.advance,label='Capture',hotkey='c')
        self.g_pool.quickbar.append(self.button)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
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
            audio.say("Capture 10 calibration patterns.")
        self.collect_new = True

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = glfwGetMonitors()[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width = 640,360

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


    def calculate(self):
        self.calculated = True
        camera_matrix, dist_coefs = _calibrate_camera(np.asarray(self.img_points),
                                                    np.asarray(self.obj_points),
                                                    (self.img_shape[1], self.img_shape[0]))
        np.save(os.path.join(self.g_pool.user_dir,'camera_matrix.npy'), camera_matrix)
        np.save(os.path.join(self.g_pool.user_dir,"dist_coefs.npy"), dist_coefs)
        np.save(os.path.join(self.g_pool.user_dir,"camera_resolution.npy"), np.array([self.img_shape[1], self.img_shape[0]]))
        audio.say("Camera calibrated. Calibration saved to user folder")
        logger.info("Camera calibrated. Calibration saved to user folder")

    def update(self,frame,recent_pupil_positions,events):
        if self.collect_new:
            img = frame.img
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

        if self._window:
            self.gl_display_in_window()

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        clear_gl_screen()
        #todo write code to display pattern.
        # r = 60.
        # gl.glMatrixMode(gl.GL_PROJECTION)
        # gl.glLoadIdentity()
        # draw_gl_point((-.5,-.5),50.)

        # p_window_size = glfwGetWindowSize(self._window)
        # # compensate for radius of marker
        # x_border,y_border = normalize((r,r),p_window_size)

        # # if p_window_size[0]<p_window_size[1]: #taller
        # #     ratio = p_window_size[1]/float(p_window_size[0])
        # #     gluOrtho2D(-x_border,1+x_border,y_border, 1-y_border) # origin in the top left corner just like the img np-array

        # # else: #wider
        # #     ratio = p_window_size[0]/float(p_window_size[1])
        # #     gluOrtho2D(-x_border,ratio+x_border,y_border, 1-y_border) # origin in the top left corner just like the img np-array

        # gluOrtho2D(-x_border,1+x_border,y_border, 1-y_border) # origin in the top left corner just like the img np-array

        # # Switch back to Model View Matrix
        # gl.glMatrixMode(gl.GL_MODELVIEW)
        # gl.glLoadIdentity()

        # for p in self.display_grid:
        #     draw_gl_point(p)
        # #some feedback on the detection state

        # # if self.detected and self.on_position:
        # #     draw_gl_point(screen_pos, 5.0, (0.,1.,0.,1.))
        # # else:
        # #     draw_gl_point(screen_pos, 5.0, (1.,0.,0.,1.))

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def get_init_dict(self):
        return {}


    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self._window:
            self.close_window()
        self.deinit_gui()


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


def _make_grid(dim=(11,4)):
    """
    this function generates the structure for an assymetrical circle grid
    centerd around 0 width=1, height scaled accordingly
    """
    x,y = range(dim[0]),range(dim[1])
    p = np.array([[[s,i] for s in x] for i in y], dtype=np.float32)
    p[:,1::2,1] += 0.5
    p = np.reshape(p, (-1,2), 'F')

    # scale height = 1
    x_scale =  1./(np.amax(p[:,0])-np.amin(p[:,0]))
    y_scale =  1./(np.amax(p[:,1])-np.amin(p[:,1]))

    p *=x_scale,x_scale/.5

    # center x,y around (0,0)
    x_offset = (np.amax(p[:,0])-np.amin(p[:,0]))/2.
    y_offset = (np.amax(p[:,1])-np.amin(p[:,1]))/2.
    p -= x_offset,y_offset
    return p


