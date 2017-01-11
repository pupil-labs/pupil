
'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
from libcpp.vector cimport vector

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
        bint isZero()

    cdef cppclass Matrix41d "Eigen::Matrix<double,4,1>": # eigen defaults to column major layout
        Matrix41d() except +
        Matrix41d(double w, double a, double b, double c)
        double * data()
        double& operator[](size_t)
        bint isZero()


    cdef cppclass Matrix4d "Eigen::Matrix<double,4,4>": # eigen defaults to column major layout
        Matrix4d() except +
        double& operator()(size_t,size_t)

    cdef cppclass Quaterniond "Eigen::Quaterniond": # eigen defaults to column major layout
        Quaterniond() except +
        Quaterniond( double w, double x, double y, double z)
        double w()
        double x()
        double y()
        double z()

cdef extern from 'common.h' namespace "" :

    #typdefs
    ctypedef Matrix31d Vector3
    ctypedef Matrix21d Vector2
    ctypedef Matrix41d Vector4


    cdef cppclass Observer "Observer": # eigen defaults to column major layout
        Observer() except +
        vector[Vector3] observations
        vector[double] pose
        int fix_rotation
        int fix_translation

cdef extern from 'ceres/rotation.h' namespace 'ceres':
    #template<typename T>
    #AngleAxisToQuaternion(T const* angle_axis, T* quaternion);
    #template<typename T>
    #QuaternionToAngleAxis(T const* quaternion, T* angle_axis);
    void AngleAxisToQuaternion(const double * angle_axis, double * quaternion);
    void QuaternionToAngleAxis(const double * quaternion, double * angle_axis);
    void AngleAxisRotatePoint(const double * angle_axis, const double * pt,double * result)

cdef extern from 'bundleCalibration.h':

    double bundleAdjustCalibration( vector[Observer]& obsevers, vector[Vector3]& points,bint fix_points)
