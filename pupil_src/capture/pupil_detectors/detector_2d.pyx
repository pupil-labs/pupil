'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

# cython: profile=False
import cv2
import numpy as np
from methods import Roi, normalize
from pyglui import ui
import glfw
from gl_utils import  adjust_gl_view, clear_gl_screen,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
from pyglui.cygl.utils import draw_gl_texture

cimport detector
from detector cimport *
from detector_utils cimport *
from coarse_pupil cimport center_surround

from cython.operator cimport dereference as deref
import math
import sys

cdef class Detector_2D:

    cdef Detector2D* thisptr
    cdef unsigned char[:,:,:] debugImage

    cdef dict detectProperties
    cdef bint windowShouldOpen, windowShouldClose
    cdef object _window
    cdef object menu
    cdef object gPool

    cdef int coarseDetectionPreviousWidth
    cdef object coarseDetectionPreviousPosition

    def __cinit__(self,g_pool = None, settings = None ):
        self.thisptr = new Detector2D()
    def __init__(self, g_pool = None, settings = None ):
        #debug window
        self._window = None
        self.windowShouldOpen = False
        self.windowShouldClose = False
        self.gPool = g_pool
        self.detectProperties = settings or {}
        self.coarseDetectionPreviousWidth = -1
        self.coarseDetectionPreviousPosition =  (0,0)
        if not self.detectProperties:
            self.detectProperties["coarse_detection"] = True
            self.detectProperties["coarse_filter_min"] = 128
            self.detectProperties["coarse_filter_max"] = 280
            self.detectProperties["intensity_range"] = 23
            self.detectProperties["blur_size"] = 5
            self.detectProperties["canny_treshold"] = 160
            self.detectProperties["canny_ration"] = 2
            self.detectProperties["canny_aperture"] = 5
            self.detectProperties["pupil_size_max"] = 200
            self.detectProperties["pupil_size_min"] = 30
            self.detectProperties["strong_perimeter_ratio_range_min"] = 0.6
            self.detectProperties["strong_perimeter_ratio_range_max"] = 1.1
            self.detectProperties["strong_area_ratio_range_min"] = 0.8
            self.detectProperties["strong_area_ratio_range_max"] = 1.1
            self.detectProperties["contour_size_min"] = 5
            self.detectProperties["ellipse_roundness_ratio"] = 0.09
            self.detectProperties["initial_ellipse_fit_treshhold"] = 4.3
            self.detectProperties["final_perimeter_ratio_range_min"] = 0.5
            self.detectProperties["final_perimeter_ratio_range_max"] = 1.0
            self.detectProperties["ellipse_true_support_min_dist"] = 3.0

    def get_settings(self):
        return self.detectProperties

    def __dealloc__(self):
      del self.thisptr

    def detect(self, frame_, user_roi, visualize, pause_video = False ):

        image_width = frame_.width
        image_height = frame_.height

        cdef unsigned char[:,::1] img = frame_.gray
        cdef Mat frame = Mat(image_height, image_width, CV_8UC1, <void *> &img[0,0] )

        cdef unsigned char[:,:,:] img_color
        cdef Mat frameColor
        cdef Mat debugImage

        if self.windowShouldOpen:
            self.open_window((image_width,image_height))
        if self.windowShouldClose:
            self.close_window()


        use_debugImage = self._window != None

        if visualize:
            img_color = frame_.img
            frameColor = Mat(image_height, image_width, CV_8UC3, <void *> &img_color[0,0,0] )

        if use_debugImage:
            debugImage_array = np.zeros( (image_height, image_width, 3 ), dtype = np.uint8 ) #clear image every frame
            self.debugImage = debugImage_array
            debugImage = Mat(image_height, image_width, CV_8UC3, <void *> &self.debugImage[0,0,0] )

        roi = Roi((0,0))
        roi.set( user_roi.get() )
        roi_x = roi.get()[0]
        roi_y = roi.get()[1]
        roi_width  = roi.get()[2] - roi.get()[0]
        roi_height  = roi.get()[3] - roi.get()[1]
        cdef int[:,::1] integral

        if self.detectProperties['coarse_detection']:
            scale = 2 # half the integral image. boost up integral
            # TODO maybe implement our own Integral so we don't have to half the image
            user_roi_image = frame_.gray[user_roi.view]
            integral = cv2.integral(user_roi_image[::scale,::scale])
            coarse_filter_max = self.detectProperties['coarse_filter_max']
            coarse_filter_min = self.detectProperties['coarse_filter_min']
            bounding_box , good_ones , bad_ones = center_surround( integral, coarse_filter_min/scale , coarse_filter_max/scale )

            if visualize:
                # !! uncomment this to visualize coarse detection
                #  # draw the candidates
                # for v  in bad_ones:
                #     p_x,p_y,w,response = v
                #     x = p_x * scale + roi_x
                #     y = p_y * scale + roi_y
                #     width = w*scale
                #     cv2.rectangle( frame_.img , (x,y) , (x+width , y+width) , (0,0,255)  )

                # # draw the candidates
                for v  in good_ones:
                    p_x,p_y,w,response = v
                    x = p_x * scale + roi_x
                    y = p_y * scale + roi_y
                    width = w*scale
                    cv2.rectangle( frame_.img , (x,y) , (x+width , y+width) , (255,255,0)  )
                    #responseText = '{:2f}'.format(response)
                    #cv2.putText(frame_.img, responseText,(int(x+width*0.5) , int(y+width*0.5)), cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,255) , 1 )

                    #center = (int(x+width*0.5) , int(y+width*0.5))
                    #cv2.circle( frame_.img , center , 5 , (255,0,255) , -1  )

            x1 , y1 , x2, y2 = bounding_box
            width = x2 - x1
            height = y2 - y1
            roi_x = x1 * scale + roi_x
            roi_y = y1 * scale + roi_y
            roi_width = width*scale
            roi_height = height*scale
            roi.set((roi_x, roi_y, roi_x+roi_width, roi_y+roi_height))


        # every coordinates in the result are relative to the current ROI
        cppResultPtr =  self.thisptr.detect(self.detectProperties, frame, frameColor, debugImage, Rect_[int](roi_x,roi_y,roi_width,roi_height),  visualize , use_debugImage )

        py_result = convertTo2DPythonResult( deref(cppResultPtr), frame_ , roi )

        return py_result



    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu('Pupil Detector')
        self.menu.collapsed = True
        info = ui.Info_Text("Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "\
                                +"Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "\
                                +"Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds.")
        self.menu.append(info)
        #self.menu.append(ui.Switch('coarse_detection',self.detectProperties,label='Use coarse detection'))
        self.menu.append(ui.Slider('intensity_range',self.detectProperties,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu.append(ui.Slider('pupil_size_min',self.detectProperties,label='Pupil min',min=1,max=250,step=1))
        self.menu.append(ui.Slider('pupil_size_max',self.detectProperties,label='Pupil max',min=50,max=400,step=1))
        self.menu.append(ui.Button('Open debug window',self.toggle_window))
        #advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        #advanced_controls_menu.append(ui.Slider('contour_size_min',self.detectProperties,label='Contour min length',min=1,max=200,step=1))
        #advanced_controls_menu.append(ui.Slider('ellipse_true_support_min_dist',self.detectProperties,label='ellipse_true_support_min_dist',min=0.1,max=7,step=0.1))
        #self.menu.append(advanced_controls_menu)
        sidebar.append(self.menu)

    def deinit_gui(self):
        self.gPool.sidebar.remove(self.menu)
        self.menu = None

    def toggle_window(self):
        if self._window:
            self.windowShouldClose = True
        else:
            self.windowShouldOpen = True

    def open_window(self,size):
        if not self._window:
            if 0: #we are not fullscreening
                monitor = glfw.glfwGetMonitors()[self.monitor_idx]
                mode = glfw.glfwGetVideoMode(monitor)
                width, height= mode[0],mode[1]
            else:
                monitor = None
                width, height = size

            active_window = glfw.glfwGetCurrentContext()
            self._window = glfw.glfwCreateWindow(width, height, "Pupil Detector Debug Window", monitor=monitor, share=active_window)
            if not 0:
                glfw.glfwSetWindowPos(self._window,200,0)

            self.on_resize(self._window,width, height)

            #Register callbacks
            glfw.glfwSetWindowSizeCallback(self._window,self.on_resize)
            # glfwSetKeyCallback(self._window,self.on_key)
            glfw.glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            glfw.glfwMakeContextCurrent(self._window)
            basic_gl_setup()

            # refresh speed settings
            glfw.glfwSwapInterval(0)

            glfw.glfwMakeContextCurrent(active_window)

            self.windowShouldOpen = False

    # window calbacks
    def on_resize(self,window,w,h):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfw.glfwMakeContextCurrent(active_window)

    def on_close(self,window):
        self.windowShouldClose = True

    def close_window(self):
        if self._window:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwDestroyWindow(self._window)
            self._window = None
            self.windowShouldClose = False
            glfw.glfwMakeContextCurrent(active_window)

    def gl_display_in_window(self,img):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(self._window)
        clear_gl_screen()
        # gl stuff that will show on your plugin window goes here
        make_coord_system_norm_based()
        draw_gl_texture(img,interpolation=False)
        glfw.glfwSwapBuffers(self._window)
        glfw.glfwMakeContextCurrent(active_window)

    def cleanup(self):
        self.close_window() # if we change detectors, be sure debug window is also closed
        self.deinit_gui()

    def visualize(self):
        #display the debug image in the window
        if self._window:
            self.gl_display_in_window(self.debugImage)
