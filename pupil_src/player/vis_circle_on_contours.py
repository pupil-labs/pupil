'''
(*)~----------------------------------------------------------------------------------

Find approximated contours
Draw convex parent contours of inner most contours 
if  gaze_point  is inside contour:
    draw circle a
else
    draw circle b

tested on black concentric circles on white background,
enviroment iluminated by fluorecent (127v 30watt) unclosed lamp on 4m high ceiling

Author: Carlos Picanco.
Hack of plugin.py and vis_circle.py from Pupil - eye tracking platform (v0.3.7.4)
    
----------------------------------------------------------------------------------~(*)
'''


from glfw import *
from plugin import Plugin
from ctypes import c_int,c_float,c_bool
from gl_utils import adjust_gl_view, clear_gl_screen, draw_gl_texture, draw_gl_polyline

from methods import denormalize
from player_methods import transparent_circle

import numpy as np
import OpenGL.GL as gl
import cv2, logging, platform 
logger = logging.getLogger(__name__)


# Each frame has an index (ID) hierarchy tree defined by the cv2.RETR_TREE from cv2.findContours
# this is just to remember that its a tree
_RETR_TREE = 0

# Constants for the hierarchy[_RETR_TREE][contour][{next,back,child,parent}]
_ID_NEXT = 0
_ID_BACK = 1
_ID_CHILD = 2
_ID_PARENT = 3

# Channel constants
_CH_B = 0  
_CH_G = 1
_CH_R = 2
_CH_0 = 3

def find_edges(img, threshold, cv2_thresh_mode):
    blur = cv2.GaussianBlur(img,(5,5),0)
    edges = []
    for gray in (blur[:,:,_CH_B], blur[:,:,_CH_G], blur[:,:,_CH_R]):
        if threshold == 0: # its not finished...
            edg = cv2.Canny(gray, 0, 50, apertureSize = 5)
            edg = cv2.dilate(edg, None)
            edges.append(edg)
        else:
            retval, edg = cv2.threshold(gray, threshold, 255, cv2_thresh_mode)
            edges.append(edg)
    return edges

def is_circle(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    radius = w / 2
    isc = abs(1 - (w / h)) <= 1 and abs(1 - (area / (np.pi * pow(radius, 2)))) <= 20 #20, adjusted by experimentation
    return isc

def idx(depth, prime, hierarchy): #get_contour_id_from_depth
    if not depth == 0:
        next_level = hierarchy[_RETR_TREE][prime][_ID_PARENT]
        return idx(depth -1, next_level, hierarchy)
    else:
        return hierarchy[_RETR_TREE][prime][_ID_PARENT]  

#approximate and draw contour by its index
def draw_approx(img, index, contours, y, detection_color):
    cnt = contours[index]
    epsilon = y * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if cv2.isContourConvex(approx):
        cv2.drawContours(img,[approx],0 ,detection_color[:3],1) 
        return approx
    else:
        approx = []
        return approx

#get and draw contours
def draw_contours(img, contours, hierarchy, y, detection_color):    
    form_contours = []
    for i, cnt in enumerate(contours):
        epsilon = y * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        if hierarchy[_RETR_TREE][i][_ID_CHILD] == -1: # if the contour has no child
            if cv2.isContourConvex(approx): 
                if len(approx) > 5:
                    if is_circle(approx) and cv2.contourArea(approx) > 1000:
                        approx = draw_approx(img, idx(2, i, hierarchy), contours, y, detection_color)
                        if len(approx) > 0: 
                            form_contours.append(approx)
    return form_contours
    
# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)


class Circle_on_Contours(Plugin):
    def __init__(
        self, g_pool = None,
        radius = 20,
        color1 = (1.,.2,.4,.5), color2 = (1.,.2,.1,.5), color3 = (0.1,.2,.1,.5),
        thickness = 1,
        full = False,
        epsilon = 0.007,
        blackbkgnd = False):
        super(Circle_on_Contours, self).__init__()
        # pupil standards
        self.g_pool = g_pool 
        self.order = .5

        # circle
        self.radius = c_int(int(radius))
        self.color1 = (c_float * 4)(*color1)
        self.color2 = (c_float * 4)(*color2)
        self.thickness = c_int(int(thickness))
        self.full = c_bool(bool(full))

        # world window
        self.blackbkgnd = c_bool(bool(blackbkgnd))

        # plugin window
        self.window_should_open = False
        self.window_should_close = False
        self._window = None
        self.fullscreen = c_bool(0)
        self.monitor_idx = c_int(0)
        self.monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]

        # detector
        self.candraw = False
        self.detection_color = (c_float*4)(*color3)
        self.detection_method_idx = c_int(0)
        self.detection_method_names = ['Method 1', 'Method 2']
        self.threshold = c_int(255)
        self.channelidx = c_int(6)
        self.channel_names = ['B', 'G', 'R', 'GR', 'BR', 'BG', 'BGR']
        self.epsilon = c_float(epsilon)

        logger.info('Circle_on_Contours plugin initialization on ' + platform.system())
        #primary_monitor = glfwGetPrimaryMonitor()

    def init_gui(self,atb_pos=(630, 10)):
        import atb

        # creating an AntTweakBar.
        atb_label = "Circle on Contours"
        self._bar = atb.Bar(
            name = self.__class__.__name__,
            label = atb_label,
            help = "ref detection parameters",
            color = (50, 50, 50),
            alpha = 100,
            text = 'light',
            position = atb_pos,
            refresh = .3,
            size = (300, 300))

        # circle parameters
        self._bar.add_var('circle in color',self.color1)
        self._bar.add_var('circle out color',self.color2)
        self._bar.add_var('circle radius',self.radius)
        self._bar.add_var('circle thickness',self.thickness, min = 0)
        self._bar.add_var('circle filled',self.full)
        self._bar.add_separator('sep1')

        # detector
        self._bar.add_var('contour color',self.detection_color)
        self._bar.add_var('black background', self.blackbkgnd)
        self._bar.add_var('threshold', self.threshold, min = 0, max = 255, step = 1)
        self._bar.add_var('Epsilon',self.epsilon, min = 0, max = 5, step = 0.0001)

        detection_method = atb.enum("Method",dict(((key,val) for val,key in enumerate(self.detection_method_names)))) 
        self._bar.add_var("Detection Method",self.detection_method_idx, vtype = detection_method)
        
        edges_channel = atb.enum("Channel",dict(((key,val) for val,key in enumerate(self.channel_names)))) 
        self._bar.add_var("Channels",self.channelidx, vtype = edges_channel)

        self._bar.add_separator('sep2')

        # window parameters
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        self._bar.add_var("wnd monitor",self.monitor_idx, vtype = monitor_enum)
        self._bar.add_var("wnd fullscreen", self.fullscreen)
        self._bar.add_button('remove plugin',self.unset_alive)
        self._bar.add_button("  show window", self.do_open, key = 's')
        self._bar.add_button("  close window", self.close_window, key = 'c')

    def unset_alive(self):
        self.alive = False

    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = self.monitor_handles[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 1280,720

            self._window = glfwCreateWindow(height, width, "Plugin Window", monitor=monitor, share=None)
            if not self.fullscreen.value:
                glfwSetWindowPos(self._window,200,0)

            on_resize(self._window,height,width)

            #Register callbacks
            glfwSetWindowSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_plugin_window_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            gl.glEnable(gl.GL_POINT_SMOOTH)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            gl.glClearColor(1.,1.,1.,0.)

            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)
            self.window_should_open = False


    def on_plugin_window_key(self,window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    self.on_close()

    def on_close(self,window = None):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

    def update(self,frame,recent_pupil_positions,events):
        #logger.debug(len(frame.img))

        if self.detection_method_idx.value == 0:
            '''

            self.edgesidx.value['B', 'G', 'R', 'GR', 'BR', 'BG', 'BGR']
            
            edges[_CH_B, _CH_G, _CH_R, _CH_0]

            img.shape[height, width, channels]


            '''

            img = frame.img
            height = img.shape[0] 
            width = img.shape[1]

            edges = find_edges(img, self.threshold.value,cv2.THRESH_OTSU)
            edges.append(np.zeros((height, width,1), np.uint8))

            # logger.debug(str(cv2.cv.GetImage(cv2.cv.fromarray(frame.img)).depth)) # 8
            #logger.debug(str(frame.img.))

            if self.channelidx.value == 6:
                
                edges_edt = cv2.max(edges[_CH_B], edges[_CH_G])
                edges_edt = cv2.max(edges_edt, edges[_CH_R])
                
                frame.img = cv2.merge([edges_edt, edges_edt, edges_edt])

            elif self.channelidx.value == 5:
                edges_edt = cv2.max(edges[_CH_B], edges[_CH_G]) 
                frame.img = cv2.merge([edges_edt, edges_edt, edges[_CH_0]])

            elif self.channelidx.value == 4:
                edges_edt = cv2.max(edges[_CH_B], edges[_CH_R])
                frame.img = cv2.merge([edges_edt, edges[_CH_0], edges_edt])

            elif self.channelidx.value == 3:
                edges_edt = cv2.max(edges[_CH_G], edges[_CH_R])
                frame.img = cv2.merge([edges[_CH_0],edges_edt, edges_edt])

            elif self.channelidx.value == _CH_R:
                edges_edt = edges[_CH_R]
                frame.img = cv2.merge([ edges[_CH_0], edges[_CH_0], edges_edt ])

            elif self.channelidx.value == _CH_G:
                edges_edt = edges[_CH_G]
                frame.img = cv2.merge([ edges[_CH_0], edges_edt, edges[_CH_0] ])  

            elif self.channelidx.value == _CH_B:
                edges_edt = edges[_CH_B]
                frame.img = cv2.merge([ edges_edt, edges[_CH_0], edges[_CH_0] ]) 

            # error: (-210) [Start]FindContours support only 8uC1 and 32sC1 images in function cvStartFindContours
            contours,hierarchy = cv2.findContours(edges_edt, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            self.contours = contours

            color1 = map(lambda x: int(x * 255),self.detection_color)
            color1 = color1[:3][::-1] + color1[-1:]
            
            contours = draw_contours(frame.img, contours, hierarchy, self.epsilon.value, color1)

            color1 = map(lambda x: int(x * 255),self.color1)
            color1 = color1[:3][::-1] + color1[-1:]
            color2 = map(lambda x: int(x * 255),self.color2)
            color2 = color2[:3][::-1] + color2[-1:]

            if self.full.value:
                thickness= -1
            else:
                thickness = self.thickness.value

            radius = self.radius.value
            pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y = True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]

            for contour in contours:
                for pt in pts:
                    Inside = cv2.pointPolygonTest(contour, pt, False)
                    if Inside > -1:
                        transparent_circle(frame.img, pt, radius = 10, color = color1, thickness = thickness)
                    else:
                        transparent_circle(frame.img, pt, radius = radius, color = color2, thickness = thickness)
            

        elif self.detection_method_idx == 1:
            pass      
        
        
        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()
        

    def gl_display(self):
        """
        use gl calls to render on world window
        """

        # gl stuff that will show on the world window goes here:

        if self._window:
            self.gl_display_in_window()

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        # draw_gl_texture(self.frame)

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def get_init_dict(self):
        return {
            'radius':self.radius.value,
            'color1':self.color1[:],'color2':self.color2[:],'color3':self.detection_color[:],
            'thickness':self.thickness.value,
            'full':self.full.value,
            'epsilon': self.epsilon.value,
            'blackbkgnd': self.blackbkgnd.value}

    def clone(self):
        return Circle_on_Contours(**self.get_init_dict())

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self._window:
            self.close_window()
        self._bar.destroy()
