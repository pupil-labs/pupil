import os, sys, platform
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from calibration_routines.optimization_calibration import point_line_calibration

import numpy as np
import math
from numpy import array, cross, dot, double, hypot, zeros
from math import acos, atan2, cos, pi, sin, radians


def quat2angle_axis(quat, identity_thresh=None):
    ''' Convert quaternion to rotation of angle around axis
    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion
    identity_thresh : None or scalar, optional
       threshold below which the norm of the vector part of the
       quaternion (x, y, z) is deemed to be 0, leading to the identity
       rotation.  None (the default) leads to a threshold estimated
       based on the precision of the input.
    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs
    Examples
    --------
    >>> theta, vec = quat2angle_axis([0, 1, 0, 0])
    >>> np.allclose(theta, np.pi)
    True
    >>> vec
    array([ 1.,  0.,  0.])
    If this is an identity rotation, we return a zero angle and an
    arbitrary vector
    >>> quat2angle_axis([1, 0, 0, 0])
    (0.0, array([ 1.,  0.,  0.]))
    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity
    rotation.  In this case we return a 0 angle and an  arbitrary
    vector, here [1, 0, 0]
    '''
    w, x, y, z = quat
    vec = np.asarray([x, y, z])
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(vec.dtype).eps * 3
        except ValueError:  # integer type
            identity_thresh = FLOAT_EPS * 3
    n = math.sqrt(x*x + y*y + z*z)
    if n < identity_thresh:
        # if vec is nearly 0,0,0, this is an identity rotation
        return 0.0, np.array([1.0, 0, 0])
    return 2 * math.acos(w), vec / n

def angle_axis2quat(theta, vector, is_normalized=False):
    ''' Quaternion for rotation of angle `theta` around `vector`
    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False
    Returns
    -------
    quat : 4 element sequence of symbols
       quaternion giving specified rotation
    Examples
    --------
    >>> q = angle_axis2quat(np.pi, [1, 0, 0])
    >>> np.allclose(q, [0, 1, 0,  0])
    True
    Notes
    -----
    Formula from http://mathworld.wolfram.com/EulerParameters.html
    '''
    vector = np.array(vector)
    if not is_normalized:
        # Cannot divide in-place because input vector may be integer type,
        # whereas output will be float type; this may raise an error in
        # versions of numpy > 1.6.1
        vector = vector / math.sqrt(np.dot(vector, vector))
    t2 = theta / 2.0
    st2 = math.sin(t2)
    return np.concatenate(([math.cos(t2)],
                           vector * st2))




def almost_equal(a, b, accuracy = 10e-6 ):
    return abs(a - b) < accuracy

if __name__ == '__main__':

#     initial_orientation = angle_axis2quat( 0 , (0.0,1.0,0.0) )
#     initial_translation = (0,0,0)


#     sphere_position = [0.0,0.0,0.0]
#     ref_points_3D = [
#         [0.0,0.0,200.0],
#         ]
#     gaze_directions_3D = [
#         [0.0,1.0,1.0],
#    ]
#     orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation)
#     angle_axis =  quat2angle_axis(orientation)
#     print angle_axis
#     assert (angle_axis[1] == [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


# ##################


#     sphere_position = [0.0,0.0,0.0]
#     ref_points_3D = [
#         [0.0,0.0,100.0],
#         [0.0,0.0,100.0],
#         [0.0,0.0,100.0]
#     ]
#     gaze_directions_3D = [
#         [-0.0,0.5,0.5],
#         [0.0,0.5,0.5],
#         [+0.0,0.5,0.5]
#     ]
#     orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation)
#     angle_axis =  quat2angle_axis(orientation)
#     assert (angle_axis[1] == [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


# # ######################


#     sphere_position = [0.0,0.0,0.0]
#     ref_points_3D = [
#         [0.0,0.0,100.0],
#         [0.0,0.0,100.0],
#         [0.0,0.0,100.0]
#     ]
#     gaze_directions_3D = [
#         [-0.5,0.5,0.5],
#         [0.0,0.5,0.5],
#         [+0.5,0.5,0.5]
#     ]
#     orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation)
#     angle_axis =  quat2angle_axis(orientation)
#     assert almost_equal(angle_axis[1] , [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


# # ######################


#     sphere_position = [0.0,0.0,0.0]
#     ##different distance shouln't make any difference
#     ref_points_3D = [
#         [0.0,0.0,10.0],
#         [0.0,0.0,100.0],
#         [0.0,0.0,1.0]
#     ]
#     gaze_directions_3D = [
#         [-0.5,0.5,0.5],
#         [0.0,0.5,0.5],
#         [+0.5,0.5,0.5]
#     ]
#     orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation)
#     angle_axis =  quat2angle_axis(orientation)
#     assert almost_equal(angle_axis[1] , [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


######################

    initial_orientation = angle_axis2quat( 0.0 , (0.0,-1.0,0.0) )
    print initial_orientation
    initial_translation = (0,0,0)

    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.0,1.0],
        #[0.0,0.0,1.0],
        #[0.0,0.0,1.0]
    ]
    gaze_directions_3D = [
        [ 1.0,0.0, 0.0],
        #[-1.0,0.0,0.0],
        #[-1.0,0.0,0.0]
    ]
    orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation)
    print orientation
    angle_axis =  quat2angle_axis(orientation)
    print angle_axis
    assert almost_equal(angle_axis[1] , [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )

    print 'Test Ended'


