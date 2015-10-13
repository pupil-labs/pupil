
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
PyObservation = namedtuple('Observation' , 'ellipse_center, ellipse_major_radius, ellipse_minor_radius, ellipse_angle,params_theta, params_psi, params_radius, circle_center, circle_normal, circle_radius')

cimport detector
from detector cimport *
from cython.operator cimport dereference as deref


cdef class Detector_3D:

    cdef Detector2D* detector_2d_ptr
    cdef EyeModelFitter *detector_3d_ptr

    cdef dict detect_properties
    cdef object menu
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
        self.debug_visualizer_3d = Visualizer(879.193)
        self.g_pool = g_pool
        self.detect_properties = settings or {}


        if not self.detect_properties:
            self.detect_properties["coarse_detection"] = True
            self.detect_properties["coarse_filter_min"] = 100
            self.detect_properties["coarse_filter_max"] = 400
            self.detect_properties["intensity_range"] = 17
            self.detect_properties["blur_size"] = 1
            self.detect_properties["canny_treshold"] = 159
            self.detect_properties["canny_ration"] = 2
            self.detect_properties["canny_aperture"] = 5
            self.detect_properties["pupil_size_max"] = 150
            self.detect_properties["pupil_size_min"] = 40
            self.detect_properties["strong_perimeter_ratio_range_min"] = 0.8
            self.detect_properties["strong_perimeter_ratio_range_max"] = 1.1
            self.detect_properties["strong_area_ratio_range_min"] = 0.6
            self.detect_properties["strong_area_ratio_range_max"] = 1.1
            self.detect_properties["contour_size_min"] = 5
            self.detect_properties["ellipse_roundness_ratio"] = 0.3
            self.detect_properties["initial_ellipse_fit_treshhold"] = 1.8
            self.detect_properties["final_perimeter_ratio_range_min"] = 0.6
            self.detect_properties["final_perimeter_ratio_range_max"] = 1.2

    def get_settings(self):
        return self.detect_properties

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



    def detect(self, frame, user_roi, visualize ):

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

        if self.detect_properties['coarse_detection']:
            scale = 2 # half the integral image. boost up integral
            # TODO maybe implement our own Integral so we don't have to half the image
            user_roi_image = frame.gray[user_roi.view]
            integral = cv2.integral(user_roi_image[::scale,::scale])
            coarse_filter_max = self.detect_properties['coarse_filter_max']
            coarse_filter_min = self.detect_properties['coarse_filter_min']
            p_x,p_y,p_w,p_response = center_surround( integral, coarse_filter_min/scale , coarse_filter_max/scale )
            roi_x = p_x * scale + roi_x
            roi_y = p_y * scale + roi_y
            roi_width = p_w*scale
            roi_height = p_w*scale
            roi.set((roi_x, roi_y, roi_x+roi_width, roi_y+roi_width))

        # every coordinates in the result are relative to the current ROI
        cpp_result_ptr =  self.detector_2d_ptr.detect(self.detect_properties, cv_image, cv_image_color, debug_image, Rect_[int](roi_x,roi_y,roi_width,roi_height), visualize , False ) #we don't use debug image in 3d model
        deref(cpp_result_ptr).timestamp = frame.timestamp

        cdef Detector_2D_Results cpp_result = deref(cpp_result_ptr)

        py_result = self.convertToPythonResult( cpp_result, frame, roi )

        ######### 3D Model Part ############

        self.detector_3d_ptr.add_observation( cpp_result_ptr, image_width, image_height , True  ) # if true is set, add_conversation converts the data to the coord space of the eye fitter
        self.detector_3d_ptr.unproject_last_contour()

        if py_result['confidence'] > 0.8:
            if self.detector_3d_ptr.pupils.size() > 3:
                pupil_radius = 1
                eye_z = 20
                self.detector_3d_ptr.unproject_observations(pupil_radius, eye_z)
                self.detector_3d_ptr.initialise_model()
                #now we have an updated eye model
                #use it to unproject contours
                #calculate uv coords of unprojected contours
                #self.detector_3d_ptr.unwrap_contours()

        if self.debug_visualizer_3d._window:
            eye = self.detector_3d_ptr.eye
            py_eye = ((eye.center[0],eye.center[1],eye.center[2]),eye.radius)
            self.debug_visualizer_3d.update_window( self.g_pool, image_width, image_height, py_eye, self.get_last_observations(5), self.get_last_unprojected_contours() )


        return py_result


    def cleanup(self):
        self.debug_visualizer_3d.close_window() # if we change detectors, be sure debug window is also closed


    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu('Pupil Detector')
        info = ui.Info_Text("Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "\
                                +"Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "\
                                +"Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds.")
        self.menu.append(info)
        self.menu.append(ui.Slider('intensity_range',self.detect_properties,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu.append(ui.Slider('pupil_size_min',self.detect_properties,label='Pupil min',min=1,max=250,step=1))
        self.menu.append(ui.Slider('pupil_size_max',self.detect_properties,label='Pupil max',min=50,max=400,step=1))
        self.menu.append(ui.Slider('ellipse_roundness_ratio',self.detect_properties,min=0.01,max=1.0,step=0.01))
        self.menu.append(ui.Slider('initial_ellipse_fit_treshhold',self.detect_properties,min=0.01,max=3.0,step=0.01))
        self.menu.append(ui.Button('Reset 3D Model', self.reset_3D_Model ))
        advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        advanced_controls_menu.append(ui.Switch('coarse_detection',self.detect_properties,label='Use coarse detection'))
        #advanced_controls_menu.append(ui.Slider('contour_size_min',self.detect_properties,label='Contour min length',min=1,max=200,step=1))

        advanced_controls_menu.append(ui.Button('Open debug window',self.toggle_window))
        self.menu.append(advanced_controls_menu)
        sidebar.append(self.menu)

    def deinit_gui(self):
        self.g_pool.sidebar.remove(self.menu)
        self.menu = None

    def reset_3D_Model(self):
         self.detector_3d_ptr.reset()

    def toggle_window(self):
        if not self.debug_visualizer_3d._window:
            self.debug_visualizer_3d.open_window()
        else:
            self.debug_visualizer_3d.close_window()


    ### Debug Helper Start ###

    def get_observation(self,index):
        cdef EyeModelFitter.Pupil p = self.detector_3d_ptr.pupils[index]
        cdef Detector_2D_Results observation = deref(p.observation)
        # returning (Ellipse, Params, Cicle). Ellipse = ([x,y],major,minor,angle). Params = (theta,psi,r)
        # Circle = (center[x,y,z], normal[x,y,z], radius)
        return PyObservation( (observation.ellipse.center[0],observation.ellipse.center[1]), observation.ellipse.major_radius,observation.ellipse.minor_radius,observation.ellipse.angle,
            p.params.theta,p.params.psi,p.params.radius,
            (p.circle.center[0],p.circle.center[1],p.circle.center[2]),
            (p.circle.normal[0],p.circle.normal[1],p.circle.normal[2]),
            p.circle.radius)

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
            p.circle.radius)

    def get_last_unprojected_contours(self):
        if self.detector_3d_ptr.pupils.size() == 0:
            return []

        cdef EyeModelFitter.Pupil p = self.detector_3d_ptr.pupils.back()
        contours = []
        for contour in p.unprojected_contours:
            c = []
            for point in contour:
                c.append([point[0],point[1],point[2]])
            contours.append(c)

        return contours

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
            p.circle.radius)


    ### Debug Helper End ###

