import os, sys, platform
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from calibration_routines.optimization_calibration import point_line_calibration


from numpy import array, cross, dot, double, hypot, zeros
from math import acos, atan2, cos, pi, sin, radians


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::

        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)

    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = zeros(3, double)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = hypot(axis[0], hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta

def almost_equal(a, b, accuracy = 10e-6 ):
    return abs(a - b) < accuracy

if __name__ == '__main__':


    sphere_position = [0.0,0.0,0.0]

    ref_points_3D = [
        [0.0,0.0,200.0],
        ]
    gaze_directions_3D = [
        [0.0,1.0,1.0],
   ]
    transformation, _ = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D )
    angle_axis = R_to_axis_angle(transformation)
    # we use almost equal because R_to_axis_angle is numerically very instable
    print angle_axis
    assert (angle_axis[0] == [1,0,0,]).all() and almost_equal(angle_axis[1] , radians(45) )

    sphere_position = [0.0,0.0,0.0]

    ref_points_3D = [
        [0.0,0.0,200.0],
        [0.0,0.0,200.0],
        [0.0,0.0,200.0]
    ]
    gaze_directions_3D = [
        [0.0,1.0,0.01],
        [0.0,1.0,0.01],
        [0.0,1.0,0.01]
    ]
    transformation, _ = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D )
    angle_axis = R_to_axis_angle(transformation)
    print angle_axis
    # we use almost equal because R_to_axis_angle is numerically very instable
    assert (angle_axis[0] == [1,0,0,]).all() and almost_equal(angle_axis[1] , radians(90) )

