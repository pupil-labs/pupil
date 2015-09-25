
cimport numpy as np


cdef extern from '<opencv2/core/types_c.h>':

  int CV_8UC1
  int CV_8UC3


cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Mat':

  cdef cppclass Mat :
      Mat() except +
      Mat( int height, int width, int type, void* data  ) except+

cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Rect':

  cdef cppclass Rect_[T]:
    Rect_() except +
    Rect_( T x, T y, T width, T height ) except +


cdef extern from 'detect_2d.hpp':


  cdef struct Result:
    int test
    double test2

  cdef struct DetectProperties:
    int intensity_range
    int blur_size
    double canny_treshold
    double canny_ration
    int canny_aperture
    int pupil_max
    int pupil_min
    int target_size

  cdef Result detect( Mat& image, Rect_[int]& usr_roi, Mat& color_image , bint visualize , DetectProperties& prop )


cdef class Detector_2D:

      def detect(self, frame, roi, visualization ):
          width = frame.width
          height = frame.height


          cdef unsigned char[:,::1] img = frame.gray
          cdef unsigned char[:,:,:] img_color = frame.img
          cdef Mat cv_image = Mat(height, width, CV_8UC1, <void *> &img[0,0] )
          cdef Mat cv_image_color = Mat(height, width, CV_8UC3, <void *> &img_color[0,0,0] )


          detect_properties  = {};
          detect_properties["intensity_range"] = 20;
          detect_properties["blur_size"] = 1;
          detect_properties["canny_treshold"] = 159;
          detect_properties["canny_ration"] = 2;
          detect_properties["canny_aperture"] = 5;
          detect_properties["pupil_max"] = 150;
          detect_properties["pupil_min"] = 40;
          detect_properties["target_size"] = 100.0;


          x = roi.get()[0]
          y = roi.get()[1]
          width  = roi.get()[2] - roi.get()[0]
          height  = roi.get()[3] - roi.get()[1]
          result =  detect( cv_image , Rect_[int](x,y,width,height), cv_image_color, True , detect_properties)
          return result



