
'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
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

cdef extern from 'common.h':

    #typdefs
    ctypedef Matrix31d Vector3
    ctypedef Matrix21d Vector2

    cdef cppclass Observation "Observation": # eigen defaults to column major layout
        Observation() except +
        vector[Vector3] dirs
        vector[double] camera

cdef extern from 'pointLineCalibration.h':


    bint pointLineCalibration( Vector3 spherePosition,const vector[Vector3]& points,const vector[Vector3]& lines,  double[4]& orientation ,
                                 double[3]& translation, bint fixTranslation,
                                 Vector3 translationLowerBound, Vector3 translationUpperBound  )

cdef extern from 'lineLineCalibration.h':

    bint bundleAdjustCalibration( const vector[Observation]& observation, vector[Vector3]& points,
                                 vector[vector[double]]& cameras, bint fixTranslation, bint useWeight  )
