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
from math import *


cdef extern from "../../singleeyefitter/CircleGoodness3D.h" namespace "singleeyefitter":

    cdef cppclass CircleGoodness3D:
        CircleGoodness3D() except +
        double operator()( Circle& circle, Contours3D contours)

cdef extern from "../TestUtils.h" :

    vector[Vector3] createCirclePointsOnSphere( Vector2 center, double opening_angle_alpha,  int amount, float circle_segment_range, double randomAmount )


def almost_equal(a, b, accuracy = 10e-8 ):
    return abs(a - b) < accuracy

def test():

    cdef CircleGoodness3D circleTest

    #create test contours and a test circle

    #1 unit circle
    cdef Circle circle
    circle.normal = Vector3(0,1,0)
    circle.radius = 1
    circle.center = Vector3(0,0,0)


    cdef Contours3D contours

    cdef vector[Vector3] contour

    contour.push_back( Vector3(0,0,-1) )
    contour.push_back( Vector3(1,0,0) )
    contours.push_back(contour)

    goodness = circleTest(circle, contours)
    assert goodness == 0.25

    contour.clear()
    contours.clear();
    contour.push_back( Vector3(1,0,0) )
    contour.push_back( Vector3(0,0,1) )
    contour.push_back( Vector3(-1,0,0) )
    contours.push_back(contour)

    goodness = circleTest(circle, contours)
    assert goodness == 0.5

    #test going through zero
    # we need to add points near zero, because we can't detect wrap arounds if the angle is the same or bigger than pi
    contour.clear()
    contours.clear();
    contour.push_back( Vector3(0.0,0,-1) )
    contour.push_back( Vector3(1,0,-0.1) )
    contour.push_back( Vector3(1,0,0.1) )
    contour.push_back( Vector3(0.0,0,1) )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert goodness == 0.5

    #the other way around
    contour.clear()
    contours.clear();
    contour.push_back( Vector3(0.0,0,1) )
    contour.push_back( Vector3(1,0,0.1) )
    contour.push_back( Vector3(1,0,-0.1) )
    contour.push_back( Vector3(0.0,0,-1) )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert goodness == 0.5

    # if the angle is smaller than pi we can
    contour.clear()
    contours.clear();
    contour.push_back( Vector3(0.0,0,-1) )
    contour.push_back( Vector3(sin(pi/4),0,cos(pi/4)) )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert goodness == 0.375


    # test overlapping contours
    contour.clear()
    contours.clear();
    contour.push_back( Vector3(0.0,0,1) )
    contour.push_back( Vector3(-1.0,0,1) )
    contour.push_back( Vector3(0,0,-1) )
    contours.push_back(contour)

    contour.clear()
    contour.push_back( Vector3(-sin(pi/4),0,cos(pi/4)) )
    contour.push_back( Vector3(-1.0,0,0) )
    contour.push_back( Vector3(-sin(pi/4),0,-cos(pi/4)) )

    #reverse an overlapping contour
    contour.clear()
    contour.push_back( Vector3(-0.1,0,-0.9) )
    contour.push_back( Vector3(-0.8,0,0.3) )
    contour.push_back( Vector3(-0.3,0,0.6) )

    goodness = circleTest(circle, contours)
    assert goodness == 0.5


    #if points are too far away from the circle we won't consider it for goodness
    contour.clear()
    contours.clear();
    contour.push_back( Vector3(0.0,0,-1) )
    contour.push_back( Vector3(sin(pi/4),0,cos(pi/4)) )
    contour.push_back( Vector3(sin(pi/5),0,cos(pi/5)+ 0.3) ) # bad point not considerd for goodness
    contour.push_back( Vector3(sin(pi/5),0,cos(pi/5)- 0.3) ) # bad point not considerd for goodness
    contour.push_back( Vector3(sin(pi/5)+0.3,0,cos(pi/5)) ) # bad point not considerd for goodness
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert goodness == 0.375

    #check that the angle between two following points is less than pi/2
    circle_distortion =  0.0
    circle_segment_amount = 0.8
    circle_point_amount =  4
    circle_opening = pi/4.0
    #test with random points
    cdef Vector2 center ##spherical coords

    phi_circle_center = center[0] = pi/2.0
    theta_circle_center = center[1] = 0.0

    sphere_radius = 1.0
    circle_z = sphere_radius * sin(phi_circle_center) * cos(theta_circle_center)*cos(circle_opening)
    circle_x = sphere_radius * sin(phi_circle_center) * sin(theta_circle_center)*cos(circle_opening)
    circle_y = sphere_radius * cos(phi_circle_center)*cos(circle_opening)

    circle.center = Vector3(circle_x,circle_y,circle_z)
    norm = circle.center.norm()
    circle.normal = Vector3(circle.center[0]/norm, circle.center[1]/norm,circle.center[2]/norm)
    circle.radius = sin(circle_opening)

    contour = createCirclePointsOnSphere(  center, circle_opening, circle_point_amount,  circle_segment_amount , circle_distortion )
    contours.clear();
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert almost_equal( goodness, 0.8 );

    #shouldn't change if we add a second one
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert almost_equal( goodness, 0.8 );

    #shouldn't change if we add a smaller one
    circle_segment_amount = 0.3
    contour = createCirclePointsOnSphere(  center, circle_opening, circle_point_amount,  circle_segment_amount , circle_distortion )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert almost_equal( goodness, 0.8 );

    #closed one
    contours.clear();
    circle_segment_amount = 1.0
    contour = createCirclePointsOnSphere(  center, circle_opening, circle_point_amount,  circle_segment_amount , circle_distortion )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert almost_equal( goodness, 1.0 );

    #distortion shouldn't change goodness much
    #add more points s error gets smaller
    contours.clear();
    circle_segment_amount = 0.5
    circle_point_amount = 30
    circle_distortion =  0.01
    contour = createCirclePointsOnSphere(  center, circle_opening, circle_point_amount,  circle_segment_amount , circle_distortion )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert almost_equal( goodness, 0.5 , 10e-3 );

    #size shouldn't change goodness
    contours.clear();
    circle_segment_amount = 0.96
    circle_distortion =  0.0
    circle_opening = pi/4
    contour = createCirclePointsOnSphere(  center, circle_opening, circle_point_amount,  circle_segment_amount , circle_distortion )
    contours.push_back(contour)
    goodness = circleTest(circle, contours)
    assert almost_equal( goodness, 0.96 );

    print 'passed'
