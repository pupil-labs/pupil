

import numpy as np


def distance_point_line( ref_point, line ):

    p1 = line[0]
    p2 = line[1]

    distance = np.linalg.norm( np.cross((ref_point-p1),(ref_point-p2)) )  / np.linalg.norm((p2 -p1));
    return distance


if __name__ == '__main__':


    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,0.0,1.0 ])

    #find nearest point on line for ref point
    ref_point = np.array([0.0,0.5,1.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.5

    ref_point = np.array([0.0,-0.5,1.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.5

    ref_point = np.array([0.5,0.5,1.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == np.sqrt(0.5)

    ref_point = np.array([-0.5,0.5,1.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == np.sqrt(0.5)

    line_p2 = np.array([0.0,0.0,-1.0 ])
    ref_point = np.array([-0.5,0.5,1.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == np.sqrt(0.5)

    line_p2 = np.array([0.0,0.0,1.0 ])
    ref_point = np.array([ 0.0,0.0,0.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    assert distance == 0.0

    line_p1 = np.array([0.0,0.0,0.0 ])
    line_p2 = np.array([0.0,1.0,0.0 ])

    #find nearest point on line for ref point
    ref_point = np.array([0.0,0.0,100.0])
    distance =  distance_point_line( ref_point , (line_p1, line_p2) )
    print distance
    assert distance == 100.0
