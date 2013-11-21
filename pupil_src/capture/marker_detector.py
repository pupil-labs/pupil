import os
import cv2
import numpy as np
from gl_utils import draw_gl_polyline,adjust_gl_view,clear_gl_screen,draw_gl_point,draw_gl_point_norm,basic_gl_setup
from methods import normalize
import atb
import audio
from ctypes import c_int,c_bool

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

        self.rects = []

        self.aperture = c_int(9)
        self.min_marker_perimeter = 40

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

        edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.aperture.value, 9)

        contours, hierarchy = cv2.findContours(edges,
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS

        # remove extra encapsulation
        hierarchy = hierarchy[0]
        # turn outmost list into array
        contours =  np.array(contours)
        # keep only contours                        with parents     and      children
        contained_contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]
        # turn on to debug contours
        # cv2.drawContours(img, contours,-1, (0,255,255))
        # cv2.drawContours(img, contained_contours,-1, (0,0,255))
        # cv2.drawContours(img, aprox_contours,-1, (255,0,0))

        # contained_contours = contours #overwrite parent children check

        #filter out rects
        aprox_contours = [cv2.approxPolyDP(c,epsilon=2.5,closed=True) for c in contained_contours]
        # any rectagle will be made of 4 segemnts in its approximation we dont need to find a marker so small that we cannot read it in the end...
        #also we want all contours to be counter clockwise oriented, we use convex hull fot this:
        rect_cand = [cv2.convexHull(c) for c in aprox_contours if c.shape[0]==4 and cv2.arcLength(c,closed=True) > self.min_marker_perimeter]
        # a covex quadrangle is not what we are looking for.
        rect_cand = [r for r in rect_cand if r.shape[0]==4]

        # subpixel corner fitting
        rects = np.array(rect_cand,dtype=np.float32)
        rects_shape = rects.shape
        rects.shape = (-1,2) #flatten for rectsubPix
        # define the criteria to stop and refine the rects
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(gray_img,rects,(3,3),(-1,-1),criteria)
        rects.shape = rects_shape #back to old layout [[rect],[rect],[rect]...] with rect = [corner,corner,corncer,corner]

        def decode(square_img,grid):
            step = square_img.shape[0]/grid
            start = step/2
            msg = otsu[start::step,start::step]
            # border is: first row, last row, first column, last column
            if msg[0,:].any() or msg[-1:0].any() or msg[:,0].any() or msg[:,-1].any():
                # logger.debug("This is not a valid marker: \n %s" %msg)
                return None
            # strip border to get the message
            msg = msg[1:-1,1:-1]/255

            # out first bit is encoded in the orientation corners of the marker:
            #               MSB = 0                   MSB = 1
            #               W|*|*|W   ^               B|*|*|B   ^
            #               *|*|*|*  / \              *|*|*|*  / \
            #               *|*|*|*   |  UP           *|*|*|*   |  UP
            #               B|*|*|W   |               W|*|*|B   |
            # 0,0 -1,0 -1,-1, 0,-1
            # angles are counter-clockwise rotation
            corners = msg[0,0], msg[-1,0], msg[-1,-1], msg[0,-1]

            if sum(corners) == 3:
                msg_int = 0
            elif sum(corners) ==1:
                msg_int = 1
                corners = tuple([1-c for c in corners]) #simple inversion
            else:
                return None

            #read rotation of marker by now we are guaranteed to have 3w and 1b
            if corners == (0,1,1,1):
                angle = 270
            elif corners == (1,0,1,1):
                angle = 0
            elif corners == (1,1,0,1):
                angle = 90
            else:
                angle = 180

            msg = np.rot90(msg,-angle/90)
            # W  |LSB| 1 |W      ^
            # 2  | 3 | 4 |5     / \
            # 6  | 7 | 8 |9      |  UP
            # MSB| 10| 11|W      |
            # print angle
            # print msg.transpose() # we align the output of print to align with pixels that you see
            msg = msg.tolist()

            #strip orientation corners from marker
            del msg[0][0]
            del msg[0][-1]
            del msg[-1][0]
            del msg[-1][-1]
            #flatten list
            msg = [item for sublist in msg for item in sublist]
            while msg:
                # [0,1,0,1] -> int [MSB,bit,bit,...,LSB], note the MSB is definde above
                msg_int = (msg_int<<1) + msg.pop()
            return angle,msg_int

        offset = 0
        markers = []
        for r in rects:
            # cv2.polylines(img,[np.int0(r)],isClosed=True,color=(100,200,0))
            # y_slice = int(min(r[:,:,0])-1),int(max(r[:,:,0])+1)
            # x_slice = int(min(r[:,:,1])-1),int(max(r[:,:,1])+1)
            # marker_img = img[slice(*x_slice),slice(*y_slice)]
            # marker_img *= 0.1
            size = 100 # should be a multiple of marker grid
            M = cv2.getPerspectiveTransform(r,np.array(((0,0),(size,0),(size,size),(0,size)),dtype=np.float32) ) #bottom left,top left, top right, bottom right in image
            flat_marker_img =  cv2.warpPerspective(gray_img, M, (size,size) )#[, dst[, flags[, borderMode[, borderValue]]]])

            # Otsu documentation here :
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
            _ , otsu = cv2.threshold(flat_marker_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


            # cosmetics -- getting a cleaner display of the rectangle marker
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            cv2.erode(otsu,kernel,otsu, iterations=3)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # cv2.dilate(otsu,kernel,otsu, iterations=1)



            marker = decode(otsu, 5) # the full 6*6 marker
            if marker is not None:
                angle,msg = marker
                # roll points such that the marker points correspond with oriented marker
                rot_r = np.roll(r,angle/90,axis=0)
                # this way we get the matrix transform with rotation included
                marker_to_screen = cv2.getPerspectiveTransform(np.array(((1,0),(1,1),(0,1),(0,0)),dtype=np.float32),rot_r)
                screen_to_marker = cv2.getPerspectiveTransform(rot_r,np.array(((0.,0.),(0.,1),(1,1),(1,0.)),dtype=np.float32))
                #marker coord system:
                # +-----------+
                # |0,1     1,1|  ^
                # |           | / \
                # |           |  |  UP
                # |0,0     1,0|  |
                # +-----------+
                self.rects.append(r)
                img[0:flat_marker_img.shape[0],offset:flat_marker_img.shape[1]+offset,1] = np.rot90(otsu,-angle/90)
                # img[0:flat_marker_img.shape[0],offset:flat_marker_img.shape[1]+offset,2] = np.rot90(flat_marker_img,angle/90)
                centroid = [r.sum(axis=0)/4.]
                bottom_left = rot_r[0]
                center = np.array([[[0,0],[1,0],[1.5,.5],[1,1],[0,1]]],dtype=np.float32)
                center = cv2.perspectiveTransform(center,marker_to_screen)
                cv2.polylines(img,np.int0(center),color = (0,0,255),isClosed=True)
                cv2.polylines(img,np.int0(centroid),color = (255,255,0),isClosed=True)
                cv2.putText(img,'id: '+str(msg),tuple(np.int0(bottom_left)[0,:]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,100,50))
                offset += size+10
                # marker to be returned/broadcast out -- accessible to world
                marker = {'id':msg,'pts':rot_r,'marker_to_screen':marker_to_screen,'screen_to_marker':screen_to_marker}
                if offset+size > img.shape[1]:
                    break



        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

    def gl_display(self):
        """
        for debugging now
        """

        for r in self.rects:
            r.shape = 4,2 #remove encapsulation
            draw_gl_polyline(r,(0.1,1.,1.,.5))

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

