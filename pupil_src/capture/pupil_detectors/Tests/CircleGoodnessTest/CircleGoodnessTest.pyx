
cimport typ_defs
from typ_defs cimport *


cdef extern from "../../singleeyefitter/CircleGoodness3D.h" namespace "singleeyefitter":

    cdef cppclass CircleGoodness3D[Scalar]:
        CircleGoodness3D() except +
        Scalar operator()( Circle& circle, Contours3D contours)




def test():

    cdef CircleGoodness3D[double] circleTest

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

   # goodness = circleTest(circle, contours)
   # assert goodness == 0.25

    contour.clear()
    contours.clear();
    contour.push_back( Vector3(1,0,0) )
    contour.push_back( Vector3(0,0,1) )
    contour.push_back( Vector3(-1,0,0) )
    contours.push_back(contour)

   # goodness = circleTest(circle, contours)
    #assert goodness == 0.5

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
    print "goodness " , goodness
    assert goodness == 0.5

    # if the angle is smaller than pi we can
    contour.clear()
    contours.clear();
    contour.push_back( Vector3(0.0,0,-1) )
    contour.push_back( Vector3(0.7,0,0.7) )
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
    contour.push_back( Vector3(-0.7,0,0.7) )
    contour.push_back( Vector3(-1.0,0,0) )
    contour.push_back( Vector3(-0.7,0,-0.7) )

    #reverse an overlapping contour
    contour.clear()
    contour.push_back( Vector3(-0.1,0,-0.9) )
    contour.push_back( Vector3(-0.8,0,0.3) )
    contour.push_back( Vector3(-0.3,0,0.6) )

    goodness = circleTest(circle, contours)
    assert goodness == 0.5
