
# cython: profile=False
import cv2
import numpy as np
from coarse_pupil cimport center_surround
from methods import Roi, normalize
from pyglui import ui
import glfw
from gl_utils import  adjust_gl_view, clear_gl_screen,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
from pyglui.cygl.utils import draw_gl_texture
import math

from pupil_detectors.visualizer_3d import Visualizer
from collections import namedtuple
PyObservation = namedtuple('Observation' , 'ellipse_center, ellipse_major_radius, ellipse_minor_radius, ellipse_angle,params_theta, params_psi, params_radius, circle_center, circle_normal, circle_radius ')


cimport detector
from detector cimport *
from cython.operator cimport dereference as deref


cdef class Detector_3D:

    cdef Detector2D* detector_2d_ptr
    cdef EyeModelFitter *detector_3d_ptr

    cdef dict detect_properties_2d, detect_properties_3d
    cdef object menu_2d, menu_3d
    cdef object g_pool
    cdef object debug_visualizer_3d

    def __cinit__(self):
        self.detector_2d_ptr = new Detector2D()
        focal_length = 879.193
        region_band_width = 5
        region_step_epsilon = 0.5
        self.detector_3d_ptr = new EyeModelFitter(focal_length, region_band_width, region_step_epsilon)

    def __init__(self, g_pool = None, settings = None ):

        #debug window
        self.debug_visualizer_3d = Visualizer(self.detector_3d_ptr.focal_length)
        self.g_pool = g_pool
        self.detect_properties_2d = settings['2D_Settings'] if settings else {}
        self.detect_properties_3d = settings['3D_Settings'] if settings else {}

        if not self.detect_properties_2d:
            self.detect_properties_2d["coarse_detection"] = True
            self.detect_properties_2d["coarse_filter_min"] = 100
            self.detect_properties_2d["coarse_filter_max"] = 400
            self.detect_properties_2d["intensity_range"] = 17
            self.detect_properties_2d["blur_size"] = 3
            self.detect_properties_2d["canny_treshold"] = 200
            self.detect_properties_2d["canny_ration"] = 3
            self.detect_properties_2d["canny_aperture"] = 5
            self.detect_properties_2d["pupil_size_max"] = 150
            self.detect_properties_2d["pupil_size_min"] = 40
            self.detect_properties_2d["strong_perimeter_ratio_range_min"] = 0.8
            self.detect_properties_2d["strong_perimeter_ratio_range_max"] = 1.1
            self.detect_properties_2d["strong_area_ratio_range_min"] = 0.6
            self.detect_properties_2d["strong_area_ratio_range_max"] = 1.1
            self.detect_properties_2d["contour_size_min"] = 5
            self.detect_properties_2d["ellipse_roundness_ratio"] = 0.1
            self.detect_properties_2d["initial_ellipse_fit_treshhold"] = 1.8
            self.detect_properties_2d["final_perimeter_ratio_range_min"] = 0.6
            self.detect_properties_2d["final_perimeter_ratio_range_max"] = 1.2
            self.detect_properties_2d["ellipse_true_support_min_dist"] = 4.0

        if not self.detect_properties_3d:
            self.detect_properties_3d["max_fit_residual"] = 20.0
            self.detect_properties_3d["max_circle_variance"] = 1.0
            self.detect_properties_3d["pupil_radius_min"] = 2.0 # millimeters
            self.detect_properties_3d["pupil_radius_max"] = 4.0

    def get_settings(self):
        return {'2D_Settings': self.detect_properties_2d , '3D_Settings' : self.detect_properties_3d }

    def __dealloc__(self):
      del self.detector_2d_ptr
      del self.detector_3d_ptr

    cdef convertToPythonResult(self, Detector_2D_Results& result, object frame, object roi ):

        e = ((result.ellipse.center[0],result.ellipse.center[1]), (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0) , result.ellipse.angle * 180 / np.pi - 90 )
        py_result = {}
        py_result['confidence'] = result.confidence
        py_result['ellipse'] = e
        py_result['pos_in_roi'] = e[0]
        py_result['major'] = max(e[1])
        py_result['diameter'] = max(e[1])
        py_result['minor'] = min(e[1])
        py_result['axes'] = e[1]
        py_result['angle'] = e[2]
        e_img_center = roi.add_vector(e[0])
        norm_center = normalize(e_img_center,(frame.width, frame.height),flip_y=True)

        py_result['norm_pos'] = norm_center
        py_result['center'] = e_img_center
        py_result['timestamp'] = frame.timestamp
        return py_result



    def detect(self, frame, user_roi, visualize, pause = False ):

        image_width = frame.width
        image_height = frame.height


        cdef unsigned char[:,::1] img = frame.gray
        cdef Mat cv_image = Mat(image_height, image_width, CV_8UC1, <void *> &img[0,0] )

        cdef unsigned char[:,:,:] img_color
        cdef Mat cv_image_color
        cdef Mat debug_image

        if visualize:
            img_color = frame.img
            cv_image_color = Mat(image_height, image_width, CV_8UC3, <void *> &img_color[0,0,0] )


        roi = Roi((0,0))
        roi.set( user_roi.get() )
        roi_x = roi.get()[0]
        roi_y = roi.get()[1]
        roi_width  = roi.get()[2] - roi.get()[0]
        roi_height  = roi.get()[3] - roi.get()[1]
        cdef int[:,::1] integral

        if self.detect_properties_2d['coarse_detection']:
            scale = 2 # half the integral image. boost up integral
            # TODO maybe implement our own Integral so we don't have to half the image
            user_roi_image = frame.gray[user_roi.view]
            integral = cv2.integral(user_roi_image[::scale,::scale])
            coarse_filter_max = self.detect_properties_2d['coarse_filter_max']
            coarse_filter_min = self.detect_properties_2d['coarse_filter_min']
            p_x,p_y,p_w,p_response = center_surround( integral, coarse_filter_min/scale , coarse_filter_max/scale )
            roi_x = p_x * scale + roi_x
            roi_y = p_y * scale + roi_y
            roi_width = p_w*scale
            roi_height = p_w*scale
            roi.set((roi_x, roi_y, roi_x+roi_width, roi_y+roi_width))

        # every coordinates in the result are relative to the current ROI
        cpp_result_ptr =  self.detector_2d_ptr.detect(self.detect_properties_2d, cv_image, cv_image_color, debug_image, Rect_[int](roi_x,roi_y,roi_width,roi_height), visualize , False ) #we don't use debug image in 3d model
        deref(cpp_result_ptr).timestamp = frame.timestamp

        cdef Detector_2D_Results cpp_result = deref(cpp_result_ptr)

        py_result = self.convertToPythonResult( cpp_result, frame, roi )

        ######### 3D Model Part ############

        self.detector_3d_ptr.update( cpp_result_ptr )

        self.detector_3d_ptr.unproject_observation_contours( deref(cpp_result_ptr).contours )
        #self.detector_3d_ptr.unproject_last_raw_edges()
        min_radius = self.detect_properties_3d['pupil_radius_min']
        max_radius = self.detect_properties_3d['pupil_radius_max']
        max_residual = self.detect_properties_3d['max_fit_residual']
        max_variance = self.detect_properties_3d['max_circle_variance']
        self.detector_3d_ptr.fit_circle_for_eye_contours(max_residual, max_variance, min_radius, max_radius)

        if self.debug_visualizer_3d._window:
            eye = self.detector_3d_ptr.eye
            py_eye = ((eye.center[0],eye.center[1],eye.center[2]),eye.radius)
            self.debug_visualizer_3d.update_window( self.g_pool, image_width, image_height, py_eye, self )


        return py_result


    def cleanup(self):
        self.debug_visualizer_3d.close_window() # if we change detectors, be sure debug window is also closed


    def init_gui(self,sidebar):
        self.menu_2d = ui.Growing_Menu('Pupil Detector 2D')
        info = ui.Info_Text("Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "\
                                +"Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "\
                                +"Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds.")
        self.menu_2d.append(info)
        self.menu_2d.append(ui.Slider('intensity_range',self.detect_properties_2d,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu_2d.append(ui.Slider('pupil_size_min',self.detect_properties_2d,label='Pupil min',min=1,max=250,step=1))
        self.menu_2d.append(ui.Slider('pupil_size_max',self.detect_properties_2d,label='Pupil max',min=50,max=400,step=1))
        self.menu_2d.append(ui.Slider('ellipse_roundness_ratio',self.detect_properties_2d,min=0.01,max=1.0,step=0.01))
        self.menu_2d.append(ui.Slider('initial_ellipse_fit_treshhold',self.detect_properties_2d,min=0.01,max=6.0,step=0.01))
        self.menu_2d.append(ui.Slider('canny_treshold',self.detect_properties_2d,min=1,max=1000,step=1))
        self.menu_2d.append(ui.Slider('canny_ration',self.detect_properties_2d,min=1,max=4,step=1))

        self.menu_3d = ui.Growing_Menu('Pupil Detector 3D')
        info_3d = ui.Info_Text("Open the debug window to see a visualization of the 3d pupil detection." )
        self.menu_3d.append(info_3d)
        self.menu_3d.append(ui.Button('Reset 3D Model', self.reset_3D_Model ))
        self.menu_3d.append(ui.Slider('pupil_radius_min',self.detect_properties_3d,label='Pupil min radius', min=1.0,max= 8.0,step=0.1))
        self.menu_3d.append(ui.Slider('pupil_radius_max',self.detect_properties_3d,label='Pupil max radius', min=1.0,max=8.0,step=0.1))
        self.menu_3d.append(ui.Slider('max_fit_residual',self.detect_properties_3d,label='3D fit max residual', min=0.01,max=1.0,step=0.01))
        self.menu_3d.append(ui.Slider('max_circle_variance',self.detect_properties_3d,label='3D fit max circle variance', min=0.01,max=10.0,step=0.01))



        advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        advanced_controls_menu.append(ui.Switch('coarse_detection',self.detect_properties_2d,label='Use coarse detection'))
        #advanced_controls_menu.append(ui.Slider('contour_size_min',self.detect_properties_2d,label='Contour min length',min=1,max=200,step=1))

        advanced_controls_menu.append(ui.Button('Open debug window',self.toggle_window))
        self.menu_3d.append(advanced_controls_menu)
        sidebar.append(self.menu_2d)
        sidebar.append(self.menu_3d)

    def deinit_gui(self):
        self.g_pool.sidebar.remove(self.menu_2d)
        self.g_pool.sidebar.remove(self.menu_3d)
        self.menu_2d = None
        self.menu_3d = None

    def reset_3D_Model(self):
         self.detector_3d_ptr.reset()

    def toggle_window(self):
        if not self.debug_visualizer_3d._window:
            self.debug_visualizer_3d.open_window()
        else:
            self.debug_visualizer_3d.close_window()


    ### Debug Helper Start ###

    def get_circle_fitted(self):
        center = self.detector_3d_ptr.circle_fitted.center
        radius = self.detector_3d_ptr.circle_fitted.radius
        normal = self.detector_3d_ptr.circle_fitted.normal
        return [ [center[0],center[1],center[2]], radius,  [normal[0],normal[1],normal[2]] ]

    def get_observation(self,index):
        cdef EyeModelFitter.Pupil p = self.detector_3d_ptr.pupils[index]
        cdef Detector_2D_Results observation = deref(p.observation)
        # returning (Ellipse, Params, Cicle). Ellipse = ([x,y],major,minor,angle). Params = (theta,psi,r)
        # Circle = (center[x,y,z], normal[x,y,z], radius)
        return PyObservation( (observation.ellipse.center[0],observation.ellipse.center[1]), observation.ellipse.major_radius,observation.ellipse.minor_radius,observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius )

    def get_last_observations(self,count=1):
        cdef EyeModelFitter.Pupil p
        count = min(self.detector_3d_ptr.pupils.size() , count )
        cdef Detector_2D_Results observation
        for i in xrange(self.detector_3d_ptr.pupils.size()-count,self.detector_3d_ptr.pupils.size()):
            p = self.detector_3d_ptr.pupils[i]
            observation = deref(p.observation)

            yield  PyObservation( (observation.ellipse.center[0],observation.ellipse.center[1]), observation.ellipse.major_radius,observation.ellipse.minor_radius,observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius )
    def get_last_pupil_contours(self):

        contours = []
        for contour in self.detector_3d_ptr.eye_contours:
            c = []
            for point in contour:
                c.append([point[0],point[1],point[2]])
            contours.append(c)

        return contours

    def get_last_pupil_edges(self):
        edges = []
        for point in self.detector_3d_ptr.edges:
                edges.append([point[0],point[1],point[2]])
        return edges

    def get_bin_positions(self):
        if self.detector_3d_ptr.bin_positions.size() == 0:
            return []

        positions = []
        eye_position = self.detector_3d_ptr.eye.center
        eye_radius = self.detector_3d_ptr.eye.radius
        #bins are on a unit sphere
        for point in self.detector_3d_ptr.bin_positions:
            positions.append([point[0]*eye_radius+eye_position[0],point[1]*eye_radius+eye_position[1],point[2]*eye_radius+eye_position[2]])
        return positions

    def get_last_final_circle_contour(self):
        contours = []
        for contour in self.detector_3d_ptr.final_circle_contours:
            c = []
            for point in contour:
                c.append([point[0],point[1],point[2]])
            contours.append(c)

        return contours

    def get_last_final_candidate_contour(self):

        list_contours = []
        for contours in self.detector_3d_ptr.final_candidate_contours:
            cc = []
            for contour in contours:
                c = []
                for point in contour:
                    c.append([point[0],point[1],point[2]])
                cc.append(c)
            list_contours.append(cc)

        return list_contours
    # def get_last_unwrapped_contours(self):
    #     if self.detector_3d_ptr.pupils.size() == 0:
    #         return []

    #     cdef EyeModelFitter.Pupil p = self.detector_3d_ptr.pupils.back()
    #     contours = []
    #     for contour in p.unwrapped_contours:
    #         c = []
    #         for point in contour:
    #             c.append([point[0],point[1]])
    #         contours.append(c)

    #     return contours

    def get_all_pupil_observations(self):
        cdef EyeModelFitter.Pupil p
        cdef Detector_2D_Results observation
        for p in self.detector_3d_ptr.pupils:
            observation = deref(p.observation)
            yield PyObservation( (observation.ellipse.center[0],observation.ellipse.center[1]), observation.ellipse.major_radius,observation.ellipse.minor_radius,observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius )

    ### Debug Helper End ###
