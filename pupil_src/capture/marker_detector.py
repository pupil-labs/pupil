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

class Marker_Detector(Plugin):
    """docstring

    """
    def __init__(self,g_pool,atb_pos=(0,0)):
        Plugin.__init__(self)


        self.aperture = c_int(7)
        self.window_should_open = False
        self.window_should_close = False
        self._window = None
        self.fullscreen = c_bool(0)
        self.monitor_idx = c_int(0)
        self.monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        #primary_monitor = glfwGetPrimaryMonitor()

        atb_label = "marker Detection"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="marker detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum)
        self._bar.add_var("fullscreen", self.fullscreen)
        self._bar.add_var("edge apature",self.aperture, step=2,min=3)
        self._bar.add_button("  open Window   ", self.do_open, key='c')

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


    def update(self,frame,recent_pupil_positions):
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
        # cv2.drawContours(img, contours,-1, (0,255,255))
        # cv2.drawContours(img, contained_contours,-1, (0,0,255))
        # aprox_contours = [cv2.approxPolyDP(c,epsilon=2,closed=True) for c in contained_contours]
        # squares = [c for c in aprox_contours if c.shape[0]==4]
        # cv2.drawContours(img, aprox_contours,-1, (255,0,0))

        # any rectagle will be made of 4 segemnts in its approximation
        # squares = [c for c in contained_contours if cv2.approxPolyDP(c,epsilon=2,closed=True).shape[0]==4]

        # contained_contours = contours #overwrite parent children check

        aprox_contours = [cv2.approxPolyDP(c,epsilon=2.5,closed=True) for c in contained_contours]

        rect_canditates = [c for c in aprox_contours if c.shape[0]==4]

        corners = np.array(rect_canditates,dtype=np.float32)


        # for r in corners:
        #     cv2.polylines(img,[np.int0(r)],isClosed=True,color=(0,0,200))


        corners_shape = corners.shape
        corners.shape = (-1,2) #flatten for cornerSubPix

        # subpixel corner fitting
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(gray_img,corners,(2,2),(-1,-1),criteria)

        corners.shape = corners_shape #back to old layout [[rect],[rect],[rect]...] with rect = [corner,corner,corncer,corner]

        for r in corners:
            cv2.polylines(img,[np.int0(r)],isClosed=True,color=(100,200,0))
            print cv2.getPerspectiveTransform(np.array(((0.,0.),(0.,1.),(1.,1.),(1.,0.)),dtype=np.float32), r)

        # img[res[:,3],res[:,2]] =[0,255,0]


        # cv2.drawContours(img, squares,-1, (255,0,0))


        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

    def gl_display(self):
        """
        for debugging now
        """
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


