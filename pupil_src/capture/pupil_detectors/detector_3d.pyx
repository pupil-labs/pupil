
# cython: profile=False
import cv2
import numpy as np
from coarse_pupil cimport center_surround
from methods import Roi, normalize
from pyglui import ui
import glfw
from gl_utils import  adjust_gl_view, clear_gl_screen,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
from pyglui.cygl.utils import draw_gl_texture

cdef extern from '<opencv2/core/types_c.h>':

  int CV_8UC1
  int CV_8UC3


cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Mat':

  cdef cppclass Mat :
      Mat() except +
      Mat( int height, int width, int type, void* data  ) except+
      Mat( int height, int width, int type ) except+

cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Rect':

  cdef cppclass Rect_[T]:
    Rect_() except +
    Rect_( T x, T y, T width, T height ) except +

cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Scalar':

  cdef cppclass Scalar_[T]:
    Scalar_() except +
    Scalar_( T x ) except +

cdef extern from '<Eigen/Eigen>' namespace 'Eigen':
    cdef cppclass Matrix21d "Eigen::Matrix<double,2,1>": # eigen defaults to column major layout
        Matrix21d() except +
        double * data()
        double& operator[](size_t)


cdef extern from "singleeyefitter/singleeyefitter.h" namespace "singleeyefitter":

    cdef cppclass Ellipse2D[T]:
        Ellipse2D()
        Ellipse2D(T x, T y, T major_radius, T minor_radius, T angle) except +
        Matrix21d center
        T major_radius
        T minor_radius
        T angle

cdef extern from 'detect_2d.hpp':

  cdef cppclass Result[T]:
    double confidence
    Ellipse2D[T] ellipse
    double timeStamp

  cdef struct DetectProperties:
    int intensity_range
    int blur_size
    float canny_treshold
    float canny_ration
    int canny_aperture
    int pupil_size_max
    int pupil_size_min
    float strong_perimeter_ratio_range_min
    float strong_perimeter_ratio_range_max
    float strong_area_ratio_range_min
    float strong_area_ratio_range_max
    int contour_size_min
    float ellipse_roundness_ratio
    float initial_ellipse_fit_treshhold
    float final_perimeter_ratio_range_min
    float final_perimeter_ratio_range_max


  cdef cppclass Detector2D[T]:

    Detector2D() except +
    Result detect( DetectProperties& prop, Mat& image, Mat& color_image, Mat& debug_image, Rect_[int]& usr_roi , Rect_[int]& pupil_roi, bint visualize , bint use_debug_image )

cdef class Detector_3D:

    cdef Detector2D[double]* thisptr
    cdef dict detect_properties
    cdef bint window_should_open, window_should_close
    cdef object _window
    cdef object menu
    cdef object g_pool

    def __cinit__(self):
        self.thisptr = new Detector2D[double]()
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
      del self.thisptr

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


        result =  self.thisptr.detect(self.detect_properties, cv_image, cv_image_color, debug_image, Rect_[int](x,y,width,height), Rect_[int](p_y,p_x,p_w,p_h),  visualize , use_debug_image )

        e = ((result.ellipse.center[0],result.ellipse.center[1]), (result.ellipse.minor_radius * 2.0 ,result.ellipse.major_radius * 2.0) , result.ellipse.angle * 180 / np.pi - 90 )
        pupil_ellipse = {}
        pupil_ellipse['confidence'] = result.confidence
        pupil_ellipse['ellipse'] = e
        pupil_ellipse['pos_in_roi'] = e[0]
        pupil_ellipse['major'] = max(e[1])
        pupil_ellipse['diameter'] = max(e[1])
        pupil_ellipse['minor'] = min(e[1])
        pupil_ellipse['axes'] = e[1]
        pupil_ellipse['angle'] = e[2]
        e_img_center = usr_roi.add_vector(pupil_roi.add_vector(e[0]))
        norm_center = normalize(e_img_center,(frame.width, frame.height),flip_y=True)

        pupil_ellipse['norm_pos'] = norm_center
        pupil_ellipse['center'] = e_img_center
        pupil_ellipse['timestamp'] = frame.timestamp

        return pupil_ellipse



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
