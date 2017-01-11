'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.deque cimport deque
from libc.stdint cimport int32_t

cdef extern from '<opencv2/core.hpp>':

  int CV_8UC1
  int CV_8UC3


cdef extern from '<opencv2/core.hpp>' namespace 'cv::Mat':

  cdef cppclass Mat :
      Mat() except +
      Mat( int height, int width, int type, void* data  ) except+
      Mat( int height, int width, int type ) except+

cdef extern from '<opencv2/core.hpp>' namespace 'cv::Rect':

  cdef cppclass Rect_[T]:
    Rect_() except +
    Rect_( T x, T y, T width, T height ) except +
    T x, y, width, height

cdef extern from '<opencv2/core.hpp>' namespace 'cv::Point':

  cdef cppclass Point_[T]:
    Point_() except +

cdef extern from '<opencv2/core.hpp>' namespace 'cv::Scalar':

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
        double norm()


cdef extern from "geometry/Ellipse.h" namespace "singleeyefitter":

    cdef cppclass Ellipse2D[T]:
        Ellipse2D()
        Ellipse2D(T x, T y, T major_radius, T minor_radius, T angle) except +
        Matrix21d center
        T major_radius
        T minor_radius
        T angle

cdef extern from "geometry/Sphere.h" namespace "singleeyefitter":

    cdef cppclass Sphere[T]:
        Matrix31d center
        T radius

cdef extern from "geometry/Circle.h" namespace "singleeyefitter":

    cdef cppclass Circle3D[T]:
        Matrix31d center
        Matrix31d normal
        float radius

#typdefs
ctypedef Matrix31d Vector3
ctypedef Matrix21d Vector2
ctypedef vector[vector[Vector3]] Contours3D;
ctypedef Circle3D[double] Circle;
