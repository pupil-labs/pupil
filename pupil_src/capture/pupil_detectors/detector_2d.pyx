
cimport numpy as np


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
    double timeStampe

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

  cdef cppclass Detector2D[T]:

    Detector2D() except +
    Result detect( DetectProperties& prop, Mat& image, Mat& color_image, Mat& debug_image, Rect_[int]& usr_roi , bint visualize , bint use_debug_image )

cdef class Detector_2D:

    cdef Detector2D[double]* thisptr;

    def __cinit__(self):
        self.thisptr = new Detector2D[double]()
    def __init__(self):
        pass

    def __dealloc__(self):
      del self.thisptr


    def detect(self, frame, roi, visualization ):
        width = frame.width
        height = frame.height


        cdef unsigned char[:,::1] img = frame.gray
        cdef unsigned char[:,:,:] img_color = frame.img
        cdef Mat cv_image = Mat(height, width, CV_8UC1, <void *> &img[0,0] )
        cdef Mat cv_image_color = Mat(height, width, CV_8UC3, <void *> &img_color[0,0,0] )
        cdef Mat debug_image = Mat(height, width, CV_8UC3 ) ;


        detect_properties  = {};
        detect_properties["intensity_range"] = 17;
        detect_properties["blur_size"] = 1;
        detect_properties["canny_treshold"] = 159;
        detect_properties["canny_ration"] = 2;
        detect_properties["canny_aperture"] = 5;
        detect_properties["pupil_size_max"] = 150;
        detect_properties["pupil_size_min"] = 40;
        detect_properties["strong_perimeter_ratio_range_min"] = 0.8;
        detect_properties["strong_perimeter_ratio_range_max"] = 1.1;
        detect_properties["strong_area_ratio_range_min"] = 0.6;
        detect_properties["strong_area_ratio_range_max"] = 1.1;
        detect_properties["contour_size_min"] = 60;
        detect_properties["ellipse_roundness_ratio"] = 0.3;
        detect_properties["initial_ellipse_fit_treshhold"] = 1.8 ; #changed this because distance calculation on cpp returns slightly different values

        x = roi.get()[0]
        y = roi.get()[1]
        width  = roi.get()[2] - roi.get()[0]
        height  = roi.get()[3] - roi.get()[1]
        result =  self.thisptr.detect(detect_properties, cv_image, cv_image_color, debug_image, Rect_[int](x,y,width,height), True , True)


        #return result



