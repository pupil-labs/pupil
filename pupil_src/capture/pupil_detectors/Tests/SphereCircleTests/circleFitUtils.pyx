
cimport typ_defs
from typ_defs cimport *

cdef extern from "CircleOnSphereUtils.h":

    vector[Vector3] createCirclePointsOnSphere( Vector2 center, double opening_angle_alpha, int amount, float circle_segment_range, double randomAmount )
    Vector3 test_haversine( Vector2 center, double opening_angle_alpha, int amount,float circle_segment_range, Vector3 initial_guess  )
    vector[double] test_plan_fit( Vector2 center, double opening_angle_alpha, int amount,float circle_segment_range , double randomAmount  )


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

def testHaversine(center , opening_angle_alpha , amount, circle_segment_range, initial_guess):
    cdef Vector2 center_point
    center_point[0] = center[0]
    center_point[1] = center[1]
    cdef Vector3 guess
    guess[0] = initial_guess[0]
    guess[1] = initial_guess[1]
    guess[2] = initial_guess[2]
    result = test_haversine( center_point, opening_angle_alpha, amount,circle_segment_range, guess)
    return [result[0],result[1],result[2]]


def testPlanFit(center , opening_angle_alpha , amount, circle_segment_range, randomAmount = 0.0):
    cdef Vector2 center_point
    center_point[0] = center[0]
    center_point[1] = center[1]

    result = test_plan_fit( center_point, opening_angle_alpha, amount,circle_segment_range,  randomAmount)
    #circle_center , radius,  residual
    return [result[0],result[1],result[2], result[3],result[4]]
