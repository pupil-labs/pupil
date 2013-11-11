import os
import cv2
import numpy as np
from gl_utils import draw_gl_polyline,adjust_gl_view,clear_gl_screen,draw_gl_point,draw_gl_point_norm,basic_gl_setup
from methods import normalize
import atb
import audio
from ctypes import c_int,c_bool
import OpenGL.GL as gl
from OpenGL.GLU import gluOrtho2D

from glfw import *
from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)


# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Camera_Intrinsics_Estimation(Plugin):
    """Camera_Intrinsics_Calibration
        not being an actual calibration,
        this method is used to calculate camera intrinsics.

    """
    def __init__(self,g_pool,atb_pos=(0,0)):
        Plugin.__init__(self)
        self.collect_new = False
        self.calculated = False
        self.obj_grid = _gen_pattern_grid((4, 11))
        self.img_points = []
        self.obj_points = []
        self.count = 10
        self.img_shape = None

        self.display_grid = _make_grid()


        self.window_should_open = False
        self.window_should_close = False
        self._window = None
        self.fullscreen = c_bool(0)
        self.monitor_idx = c_int(0)
        self.monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        #primary_monitor = glfwGetPrimaryMonitor()

        atb_label = "estimate camera instrinsics"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum)
        self._bar.add_var("fullscreen", self.fullscreen)
        self._bar.add_button("  show pattern   ", self.do_open, key='c')
        self._bar.add_button("  Capture Pattern", self.advance, key="SPACE")
        self._bar.add_var("patterns to capture", getter=self.get_count)

    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def get_count(self):
        return self.count

    def advance(self):
        if self.count ==10:
            audio.say("Capture 10 calibration patterns.")
        self.collect_new = True

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = self.monitor_handles[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 640,360

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


    def calculate(self):
        self.calculated = True
        camera_matrix, dist_coefs = _calibrate_camera(np.asarray(self.img_points),
                                                    np.asarray(self.obj_points),
                                                    (self.img_shape[1], self.img_shape[0]))
        np.save(os.path.join(self.g_pool.user_dir,'camera_matrix.npy'), camera_matrix)
        np.save(os.path.join(self.g_pool.user_dir,"dist_coefs.npy"), dist_coefs)
        audio.say("Camera calibrated. Calibration saved to user folder")
        logger.info("Camera calibrated. Calibration saved to user folder")

    def update(self,frame,recent_pupil_positions):
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

        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

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



    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self._window:
            self.close_window()
        self._bar.destroy()


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


