from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int32_t

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
    T x, y, width, height

cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Point':

  cdef cppclass Point_[T]:
    Point_() except +

cdef extern from '<opencv2/core/core.hpp>' namespace 'cv::Scalar':

  cdef cppclass Scalar_[T]:
    Scalar_() except +
    Scalar_( T x ) except +

cdef extern from '<Eigen/Eigen>' namespace 'Eigen':
    cdef cppclass Matrix21d "Eigen::Matrix<double,2,1>": # eigen defaults to column major layout
        Matrix21d() except +
        double * data()
        double& operator[](size_t)

    cdef cppclass Matrix31d "Eigen::Matrix<double,3,1>": # eigen defaults to column major layout
        Matrix31d() except +
        Matrix31d(double x, double y, double z)
        double * data()
        double& operator[](size_t)


cdef extern from 'singleeyefitter/common/types.h':

    cdef cppclass Ellipse2D[T]:
        Ellipse2D()
        Ellipse2D(T x, T y, T major_radius, T minor_radius, T angle) except +
        Matrix21d center
        T major_radius
        T minor_radius
        T angle

    cdef cppclass Sphere[T]:
        Matrix31d center
        T radius

    cdef cppclass Circle3D[T]:
        Matrix31d center
        Matrix31d normal
        float radius

        #typdefs
    ctypedef Matrix31d Vector3
    ctypedef Matrix21d Vector2
    ctypedef vector[vector[Vector3]] Contours3D
    ctypedef vector[Vector3] Edges3D
    ctypedef vector[vector[Point_[int]]] Contours_2D
    ctypedef vector[Point_[int]] Contour_2D
    ctypedef Circle3D[double] Circle
    ctypedef Ellipse2D[double] Ellipse

    cdef struct Detector_2D_Result:
        double confidence
        Ellipse ellipse
        Contours_2D final_contours
        Contours_2D contours
        Mat raw_edges
        Rect_[int] current_roi
        double timestamp
        int image_width;
        int image_height;

    cdef struct Detector_3D_Result:
        Circle circle
        Ellipse ellipse
        double fitGoodness
        double timestamp
        Contours3D contours
        Contours3D fittedCircleContours
        Sphere[double] sphere
        Sphere[double] initialSphere
        vector[Vector3] binPositions
        Edges3D edges


    cdef struct Detector_2D_Properties:
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
        float ellipse_true_support_min_dist

    cdef struct Detector_3D_Properties:
        float max_fit_residual
        float max_circle_variance
        float pupil_radius_min
        float pupil_radius_max
        int   combine_evaluation_max
        int   combine_depth_max



cdef extern from 'detect_2d.hpp':


  cdef cppclass Detector2D:

    Detector2D() except +
    shared_ptr[Detector_2D_Result] detect( Detector_2D_Properties& prop, Mat& image, Mat& color_image, Mat& debug_image, Rect_[int]& roi, bint visualize , bint use_debug_image )


cdef extern from "singleeyefitter/EyeModelFitter.h" namespace "singleeyefitter":


    cdef cppclass EyeModelFitter:

        cppclass PupilParams:
            float theta
            float psi
            float radius

        cppclass Observation:
            shared_ptr[const Detector_2D_Result] mObservation2D;
            pair[Circle, Circle] mUnprojectedCirclePair
            Observation( shared_ptr[const Detector_2D_Result] observation, double focalLength)


        EyeModelFitter(double focalLength )

        Detector_3D_Result update_and_detect(  shared_ptr[Detector_2D_Result]& results,  Detector_3D_Properties& prop )

        void reset()
        double getFocalLength()


        double mFocalLength
        Sphere[double] mCurrentSphere





