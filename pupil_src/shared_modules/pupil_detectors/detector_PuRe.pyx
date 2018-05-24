'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

# cython: profile=False
import cv2
import numpy as np
from methods import Roi, normalize
from plugin import Plugin
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


cdef class Detector_PuRe:

    cdef PuRe* thisptr
    cdef Pupil pupil
    cdef dict detectProperties

    cdef bint windowShouldOpen, windowShouldClose
    cdef object _window
    cdef readonly object g_pool
    cdef readonly basestring uniqueness
    cdef public object menu
    cdef public object menu_icon
    cdef readonly basestring icon_chr
    cdef readonly basestring icon_font

    def __cinit__(self,g_pool = None, settings = None ):
        self.thisptr = new PuRe()
        self.pupil = Pupil()

    def __init__(self, g_pool, settings = None):
        #super().__init__(g_pool)
        self._window = None
        self.windowShouldOpen = False
        self.windowShouldClose = False
        self.g_pool = g_pool
        self.uniqueness = 'unique'
        self.icon_font = 'pupil_icons'
        self.icon_chr = chr(0xec18)
        self.detectProperties = settings or {}
        if not self.detectProperties:
            self.detectProperties["pupil_size_max"] = 120
            self.detectProperties["pupil_size_min"] = 10

    def detect(self, frame_, user_roi, visualize, pause_video = False ):

        image_width = frame_.width
        image_height = frame_.height

        cdef unsigned char[:,::1] img = frame_.gray
        cdef Mat frame = Mat(image_height, image_width, CV_8UC1, <void *> &img[0,0] )

        
        py_result = None
        roi = Roi((0,0))
        roi.set( user_roi.get() )
        roi_x = roi.get()[0]
        roi_y = roi.get()[1]
        roi_width  = roi.get()[2] - roi.get()[0]
        roi_height  = roi.get()[3] - roi.get()[1]

        #         void run(const Mat &frame, const Rect_[int] &roi, Pupil &pupil, const float &userMinPupilDiameterPx=-1, const float &userMaxPupilDiameterPx=-1

        self.thisptr.run(frame, Rect_[int](roi_x,roi_y,roi_width,roi_height),self.pupil, self.detectProperties["pupil_size_min"], self.detectProperties["pupil_size_max"] )

        py_result = {}

        ellipse = {}
        ellipse['center'] = (self.pupil.center.x, self.pupil.center.y)
        ellipse['axes'] =  (self.pupil.minorAxis(), self.pupil.majorAxis())
        ellipse['angle'] = self.pupil.angle

        norm_center = normalize( ellipse['center'] , (image_width, image_height),flip_y=True)
        py_result['norm_pos'] = norm_center
        py_result['timestamp'] = frame_.timestamp
        py_result['method'] = 'PuRe'
        py_result['topic'] = 'pupil'
        py_result['confidence'] = self.pupil.confidence
        py_result['ellipse'] = ellipse
        py_result['diameter'] = max(ellipse['axes'])

            
        return py_result
        
    def visualize(self):
        pass

    def get_settings(self):
        return {}

    def on_resolution_change(self, *args, **kwargs):
        pass
    @property
    def pretty_class_name(self):
        return 'PuRe Detector'

    def close_window(self):
        if self._window:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwDestroyWindow(self._window)
            self._window = None
            self.windowShouldClose = False
            glfw.glfwMakeContextCurrent(active_window)
    def cleanup(self):
        self.close_window()

    def init_ui(self):
        Plugin.add_menu(self)
        self.menu.label = self.pretty_class_name
        #info = ui.Info_Text("")
        #self.menu.append(info)
        #self.menu.append(ui.Switch('coarse_detection',self.detectProperties2D,label='Use coarse detection'))
        #self.menu.append(ui.Slider('intensity_range',self.detectProperties2D,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu.append(ui.Slider('pupil_size_min',self.detectProperties,label='Pupil min',min=1,max=50,step=1))
        self.menu.append(ui.Slider('pupil_size_max',self.detectProperties,label='Pupil max',min=10,max=150,step=1))
        #self.menu.append(ui.Slider('ellipse_roundness_ratio',self.detectProperties2D,min=0.01,max=1.0,step=0.01))
        #self.menu.append(ui.Slider('initial_ellipse_fit_treshhold',self.detectProperties2D,min=0.01,max=6.0,step=0.01))
        #self.menu.append(ui.Slider('canny_treshold',self.detectProperties2D,min=1,max=1000,step=1))
        #self.menu.append(ui.Slider('canny_ration',self.detectProperties2D,min=1,max=4,step=1))
        #info_3d = ui.Info_Text("Open the debug window to see a visualization of the 3D pupil detection." )
        #self.menu.append(info_3d)
        #self.menu.append(ui.Button('Reset 3D model', self.reset_3D_Model ))
        #self.menu.append(ui.Button('Open debug window',self.toggle_window))
        # self.menu.append(ui.Slider('pupil_radius_min',self.detectProperties3D,label='Pupil min radius', min=1.0,max= 8.0,step=0.1))
        # self.menu.append(ui.Slider('pupil_radius_max',self.detectProperties3D,label='Pupil max radius', min=1.0,max=8.0,step=0.1))
        # self.menu.append(ui.Slider('max_fit_residual',self.detectProperties3D,label='3D fit max residual', min=0.00,max=0.1,step=0.0001))
        # self.menu.append(ui.Slider('max_circle_variance',self.detectProperties3D,label='3D fit max circle variance', min=0.01,max=2.0,step=0.001))
        # self.menu.append(ui.Slider('combine_evaluation_max',self.detectProperties3D,label='3D fit max combinations eval', min=500,max=50000,step=5000))
        # self.menu.append(ui.Slider('combine_depth_max',self.detectProperties3D,label='3D fit max combination depth', min=10,max=5000,step=20))
        #advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        #advanced_controls_menu.append(ui.Slider('contour_size_min',self.detectProperties2D,label='Contour min length',min=1,max=200,step=1))
        #sidebar.append(advanced_controls_menu)

    def deinit_ui(self):
        Plugin.remove_menu(self)
    
