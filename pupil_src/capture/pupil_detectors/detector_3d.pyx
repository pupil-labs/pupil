
# cython: profile=False
import cv2
import numpy as np
from coarse_pupil cimport center_surround
from methods import Roi, normalize
from pyglui import ui
import glfw
from gl_utils import  adjust_gl_view, clear_gl_screen,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
from pyglui.cygl.utils import draw_gl_texture

cimport detector
from detector cimport *

cdef class Detector_3D:

    cdef Detector2D[double]* detector_2d_ptr
    cdef EyeModelFitter *detector_3d_ptr

    cdef dict detect_properties
    cdef bint window_should_open, window_should_close
    cdef object _window
    cdef object menu
    cdef object g_pool

    def __cinit__(self):
        self.detector_2d_ptr = new Detector2D[double]()
        focal_length = 879.193
        region_band_width = 5
        region_step_epsilon = 0.5
        self.detector_3d_ptr = new EyeModelFitter(focal_length, region_band_width, region_step_epsilon)

    def __init__(self, g_pool = None, settings = None ):

        #debug window
        self._window = None
        self.window_should_open = False
        self.window_should_close = False
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

    cdef convertToPythonResult(self, Result[double] result, object frame, object usr_roi, object pupil_roi ):

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
        e_img_center = usr_roi.add_vector(pupil_roi.add_vector(e[0]))
        norm_center = normalize(e_img_center,(frame.width, frame.height),flip_y=True)

        py_result['norm_pos'] = norm_center
        py_result['center'] = e_img_center
        py_result['timestamp'] = frame.timestamp
        return py_result


    def detect(self, frame, usr_roi, visualize ):

        width = frame.width
        height = frame.height

        cdef unsigned char[:,::1] img = frame.gray
        cdef Mat cv_image = Mat(height, width, CV_8UC1, <void *> &img[0,0] )

        cdef unsigned char[:,:,:] img_color
        cdef Mat cv_image_color
        cdef Mat debug_image

        if self.window_should_open:
            self.open_window((width,height))
        if self.window_should_close:
            self.close_window()


        use_debug_image = self._window != None

        if visualize:
            img_color = frame.img
            cv_image_color = Mat(height, width, CV_8UC3, <void *> &img_color[0,0,0] )

        if use_debug_image:
            debug_image = Mat(height, width, CV_8UC3 )


        x = usr_roi.get()[0]
        y = usr_roi.get()[1]
        width  = usr_roi.get()[2] - usr_roi.get()[0]
        height  = usr_roi.get()[3] - usr_roi.get()[1]
        cdef int[:,::1] integral

        if self.detect_properties['coarse_detection']:
            scale = 2 # half the integral image. boost up integral
            # TODO maybe implement our own Integral so we don't have to half the image
            integral = cv2.integral(frame.gray[::scale,::scale])
            coarse_filter_max = self.detect_properties['coarse_filter_max']
            coarse_filter_min = self.detect_properties['coarse_filter_min']
            p_x,p_y,p_w,p_response = center_surround( integral, coarse_filter_min/scale , coarse_filter_max/scale )
            p_x *= scale
            p_y *= scale
            p_w *= scale
            p_h = p_w
        else:
            p_x = x
            p_y = y
            p_w = width
            p_h = height


        pupil_roi = Roi( (0,0))
        pupil_roi.set((p_y, p_x, p_y+p_w, p_x+p_w))


        cpp_result =  self.detector_2d_ptr.detect(self.detect_properties, cv_image, cv_image_color, debug_image, Rect_[int](x,y,width,height), Rect_[int](p_y,p_x,p_w,p_h),  visualize , use_debug_image )
        py_result = self.convertToPythonResult( cpp_result, frame, usr_roi, pupil_roi )

        return py_result

 # # GL drawing
   #      #eye sphere fitter adding
   #      if result['confidence'] > 0.8:
   #          eye_model_fitter.add_pupil_labs_observation(result, contours, (frame.width, frame.height) )
   #          if eye_model_fitter.num_observations > 3:
   #              eye_model_fitter.update_model() #this calls unproject and initialize

   #      # show the visualizer
   #      visual.update_window(g_pool,contours, eye_model_fitter, frame.width, frame.height)


    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu('Pupil Detector')
        info = ui.Info_Text("Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "\
                                +"Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "\
                                +"Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds.")
        self.menu.append(info)
        self.menu.append(ui.Slider('intensity_range',self.detect_properties,label='Pupil intensity range',min=0,max=60,step=1))
        self.menu.append(ui.Slider('pupil_size_min',self.detect_properties,label='Pupil min',min=1,max=250,step=1))
        self.menu.append(ui.Slider('pupil_size_max',self.detect_properties,label='Pupil max',min=50,max=400,step=1))

        advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        advanced_controls_menu.append(ui.Switch('coarse_detection',self.detect_properties,label='Use coarse detection'))
        #advanced_controls_menu.append(ui.Slider('contour_size_min',self.detect_properties,label='Contour min length',min=1,max=200,step=1))

        advanced_controls_menu.append(ui.Button('Open debug window',self.toggle_window))
        self.menu.append(advanced_controls_menu)
        sidebar.append(self.menu)

    def deinit_gui(self):
        self.g_pool.sidebar.remove(self.menu)
        self.menu = None

    def toggle_window(self):
        if self._window:
            self.window_should_close = True
        else:
            self.window_should_open = True

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

            self.window_should_open = False

    # window calbacks
    def on_resize(self,window,w,h):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfw.glfwMakeContextCurrent(active_window)

    def on_close(self,window):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfw.glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

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
        pass

