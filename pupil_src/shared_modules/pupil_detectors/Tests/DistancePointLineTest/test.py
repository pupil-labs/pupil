'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


import numpy as np


def distance_point_line( ref_point, line ):

    p1 = line[0]
    p2 = line[1]

    denom =  np.linalg.norm(p2-p1)
    delta = - np.dot((p1 - ref_point),(p2 - p1)) / (denom*denom);
    distance = np.linalg.norm( np.cross((ref_point-p1),(ref_point-p2)) )  / np.linalg.norm((p2 -p1));
    return distance , delta

def nearest_linepoint_to_point( ref_point, line ):

    p1 = line[0]
    p2 = line[1]

    direction = p2 - p1
    denom =  np.linalg.norm(direction)
    delta = - np.dot((p1 - ref_point),(direction)) / (denom*denom)
    point  =   p1 + direction * delta
    return point

if __name__ == '__main__':


    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,0.0,1.0 ])

    #find nearest point on line for ref point
    ref_point = np.array([0.0,0.5,1.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.5

    ref_point = np.array([0.0,-0.5,1.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.5

    ref_point = np.array([0.5,0.5,1.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == np.sqrt(0.5)

    ref_point = np.array([-0.5,0.5,1.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == np.sqrt(0.5)

    line_p2 = np.array([0.0,0.0,-1.0 ])
    ref_point = np.array([-0.5,0.5,1.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == np.sqrt(0.5)

    line_p2 = np.array([0.0,0.0,1.0 ])
    ref_point = np.array([ 0.0,0.0,0.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.0

    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,1.0,0.0 ])

    ref_point = np.array([0.0,0.0,100.0])
    distance, _ =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 100.0

    ## check for the right direction
    ## delta tells us if the point is in direction from p1 to p2 or from p2 to p1
    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,1.0,0.0 ])

    #find nearest point on line for ref point
    ref_point = np.array([0.0,0.1,1.0])
    distance, delta =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 1.0 and delta == 0.1

    ## check for the right direction
    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,0.0,1.0 ])

    #find nearest point on line for ref point
    ref_point = np.array([0.0,0.0,-1.0])
    distance, delta =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.0 and delta == -1.0


    ## check for the right direction
    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,0.0,1.0 ])

    #find nearest point on line for ref point
    ref_point = np.array([0.0,0.0,10])
    point=  nearest_linepoint_to_point( ref_point , (line_p1, line_p2) )
    assert (point == [0,0,10]).all()
    print 'Test ended.'
