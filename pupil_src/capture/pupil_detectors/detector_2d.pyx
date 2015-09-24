
cimport numpy as np


cdef extern from '<opencv2/core/types_c.h>':

  int CV_8UC1


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

  cdef Result detect( Mat& image, Rect_[int]& usr_roi, Mat& color_image , bint use_color_image )


cdef class Detector_2D:

      def detect(self, frame, roi, visualization ):
          width = frame.width
          height = frame.height


          cdef unsigned char[:,::1] img = frame.gray
          cdef Mat cv_image = Mat(height, width, CV_8UC1, <void *> &img[0,0] )

          x = roi.get()[0]
          y = roi.get()[1]
          width  = roi.get()[2] - roi.get()[0]
          height  = roi.get()[3] - roi.get()[1]
          result =  detect( cv_image , Rect_[int](x,y,width,height), Mat(), False )
          return result



