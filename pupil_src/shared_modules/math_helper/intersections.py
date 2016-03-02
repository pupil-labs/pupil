'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
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

    # Check for parallel lines
    magnitude = mag (np.cross(normalise(p1, p2), normalise(p3, p4 )) )

    if magnitude > np.finfo(np.float64).eps:

        A = p1-p3
        B = p2-p1
        C = p4-p3

        ma = ((np.dot(A, C)*np.dot(C, B)) - (np.dot(A, B)*np.dot(C, C)))/ \
             ((np.dot(B, B)*np.dot(C, C)) - (np.dot(C, B)*np.dot(C, B)))
        mb = (ma*np.dot(C, B) + np.dot(A, C))/ np.dot(C, C)

        # Calculate the point on line 1 that is the closest point to line 2
        Pa = p1 + B*ma

        # Calculate the point on line 2 that is the closest point to line 1
        Pb = p3 + C* mb

        nPoint = Pa - Pb
        # Distance between lines
        intersection_dist = np.sqrt( nPoint.dot( nPoint ))

        return Pa , Pb , intersection_dist
    else:
        return None,None,None  # parallel lines


def nearest_intersection( line0 , line1 ):
    """ Calculates the nearest intersection point, and the shortest distance of line0 and line1.
    """
    Pa, Pb ,intersection_dist =  nearest_intersection_points(line0, line1)

    if Pa and Pb and intersection_dist:
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

