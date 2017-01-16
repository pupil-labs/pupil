'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

cimport typ_defs
from typ_defs cimport *

cdef extern from "../TestUtils.h" :

    vector[Vector3] createCirclePointsOnSphere( Vector2 center, double opening_angle_alpha,  int amount, float circle_segment_range, double randomAmount )

cdef extern from "../../singleeyefitter/Fit/CircleOnSphereFit.h" namespace "singleeyefitter":

    cdef cppclass CircleOnSphereFitter[Scalar]:
            CircleOnSphereFitter( const Sphere[Scalar] sphere )  except +
            bint fit(const vector[Vector3]& points)
            Scalar calculateResidual(const vector[Vector3]& points) const
            const Circle& getCircle()

def get_circle_test_points(center , opening_angle_alpha , amount, circle_segment_range, randomAmount = 0.0 ):

    cdef Vector2 center_point
    center_point[0] = center[0]
    center_point[1] = center[1]

    cdef vector[Vector3] points
    points = createCirclePointsOnSphere( center_point, opening_angle_alpha, amount, circle_segment_range, randomAmount)
    py_points = []
    for p in points:
        #print [p[0],p[1],p[2]]
        py_points.append([p[0],p[1],p[2]])

    return py_points


def testPlanFit(sphere, center , opening_angle_alpha , amount, circle_segment_range, randomAmount = 0.0):
    cdef Vector2 center_point
    center_point[0] = center[0]
    center_point[1] = center[1]

    cdef Sphere[double] s
    s.center[0] = sphere[0][0]
    s.center[1] = sphere[0][1]
    s.center[2] = sphere[0][2]
    s.radius = sphere[1]

    cdef CircleOnSphereFitter[double]* fitter
    fitter  = new CircleOnSphereFitter[double](s)

    cdef vector[Vector3] points
    points  = createCirclePointsOnSphere(center_point, opening_angle_alpha, amount,circle_segment_range,  randomAmount)
    fitter.fit(points)
    cdef Circle circle  = fitter.getCircle()

    residual = fitter.calculateResidual(points)
    del fitter
    #circle_center , normal, radius,  residual
    return [circle.center[0],circle.center[1],circle.center[2],circle.normal[0],circle.normal[1],circle.normal[2], circle.radius, residual]
