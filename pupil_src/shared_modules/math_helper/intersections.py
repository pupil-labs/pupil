'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


import sys
import numpy as np

def nearest_intersection_points( line0 , line1 ):
    """ Calculates the two nearst points, and its distance to each other on line0 and line1.
    """

    p1 = line0[0]
    p2 = line0[1]
    p3 = line1[0]
    p4 = line1[1]

    def mag(p):
        return np.sqrt( p.dot(p) )

    def normalise(p1, p2):
        p = p2 - p1
        m = mag(p)
        if m == 0:
            return [0.0, 0.0, 0.0]
        else:
            return p/m

    d1 = normalise(p1,p2)
    d2 = normalise(p3,p4)

    diff = p1 - p3;
    a01 = -d1.dot(d2);
    b0 = diff.dot(d1);


    if np.abs(a01) < 1.0:

        # Lines are not parallel.
        det = 1.0 - a01 * a01;
        b1 = -diff.dot(d2);
        s0 = (a01 * b1 - b0) / det;
        s1 = (a01 * b0 - b1) / det;

    else:

        # Lines are parallel, select any pair of closest points.
        s0 = -b0;
        s1 = 0;


    closestPoint1 = p1 + s0 * d1;
    closestPoint2 = p3 + s1 * d2;
    dist = mag( closestPoint2 - closestPoint1 )
    return closestPoint1 , closestPoint2, dist

def nearest_intersection( line0 , line1 ):
    """ Calculates the nearest intersection point, and the shortest distance of line0 and line1.
    """
    Pa, Pb ,intersection_dist =  nearest_intersection_points(line0, line1)

    if Pa is not None:
        nPoint = Pa - Pb
        return Pb + nPoint * 0.5 , intersection_dist
    else:
        return None,None # parallel lines




def nearest_linepoint_to_point( ref_point, line ):

    p1 = line[0]
    p2 = line[1]

    direction = p2 - p1
    denom =  np.linalg.norm(direction)
    delta = - np.dot((p1 - ref_point),(direction)) / (denom*denom)
    point  =   p1 + direction * delta

    d = ref_point - point
    # Distance between lines
    intersection_dist = np.sqrt( d.dot( d ))

    return point, intersection_dist

