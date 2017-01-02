'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
import os, sys, platform
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from calibration_routines.optimization_calibration import point_line_calibration, line_line_calibration

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


def mat2quat(M):
    ''' Calculate quaternion corresponding to given rotation matrix
    Parameters
    ----------
    M : array-like
      3x3 rotation matrix
    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]
    Notes
    -----
    Method claimed to be robust to numerical errors in M
    Constructs quaternion by calculating maximum eigenvector for matrix
    K (constructed from input `M`).  Although this is not tested, a
    maximum eigenvalue of 1 corresponds to a valid rotation.
    A quaternion q*-1 corresponds to the same rotation as q; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).
    References
    ----------
    * https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090
    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True
    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q



def almost_equal(a, b, accuracy = 10e-6 ):
    return abs(a - b) < accuracy

if __name__ == '__main__':

    initial_orientation = angle_axis2quat( 0 , (0.0,1.0,0.0) )
    initial_translation = (0,0,0)


    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.0,200.0],
        ]
    gaze_directions_3D = [
        [0.0,1.0,1.0],
   ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert (angle_axis[1] == [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


# ##################


    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.0,100.0],
        [0.0,0.0,100.0],
        [0.0,0.0,100.0]
    ]
    gaze_directions_3D = [
        [-0.0,0.5,0.5],
        [0.0,0.5,0.5],
        [+0.0,0.5,0.5]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert (angle_axis[1] == [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


# # ######################


    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.0,100.0],
        [0.0,0.0,100.0],
        [0.0,0.0,100.0]
    ]
    gaze_directions_3D = [
        [-0.5,0.5,0.5],
        [0.0,0.5,0.5],
        [+0.5,0.5,0.5]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert almost_equal(angle_axis[1] , [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


# # ######################


    sphere_position = [0.0,0.0,0.0]
    ##different distance shouln't make any difference
    ref_points_3D = [
        [0.0,0.0,10.0],
        [0.0,0.0,100.0],
        [0.0,0.0,1.0]
    ]
    gaze_directions_3D = [
        [-0.5,0.5,0.5],
        [0.0,0.5,0.5],
        [+0.5,0.5,0.5]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert almost_equal(angle_axis[1] , [1,0,0,]).all() and almost_equal(angle_axis[0] , radians(45) )


#####################

    initial_orientation = angle_axis2quat( -0.01 , (0.0,1.0,0.0) )
    initial_translation = (0,0,0)

    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.0,10.0],
        [0.0,0.0,10.0],
        [0.0,0.0,10.0]
    ]
    gaze_directions_3D = [
        [ 1.0,0.0, 0.0],
        [ 1.0,0.0,0.0],
        [ 1.0,0.0,0.0]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert almost_equal(angle_axis[1] , [0,-1,0,]).all() and almost_equal(angle_axis[0] , radians(90) )


#####################


    initial_orientation = angle_axis2quat( np.pi * 0.51 , (0.0,1.0,0.0) )
    initial_translation = (0,0,0)

    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.0,10.0],
        [0.0,0.0,10.0],
        [0.0,0.0,10.0]
    ]
    gaze_directions_3D = [
        [ 0.0,0.0,-1.0],
        [ 0.0,0.0,-1.0],
        [ 0.0,0.0,-1.0]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert almost_equal(angle_axis[1] , [0,1,0,]).all() and almost_equal(angle_axis[0] , radians(180) )

####################

    initial_orientation = angle_axis2quat( np.pi * 0.55 , (0.0,1.0,0.0) )
    initial_translation = (0,0,0)

    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.5,1.0],
        [0.0,0.0,1.0],
        [0.0,-0.5,1.0]
    ]
    gaze_directions_3D = [
        [ 0.0,0.5,-1.0],
        [ 0.0,0.0,-1.0],
        [ 0.0,-0.5,-1.0]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert almost_equal(angle_axis[1] , [0,1,0,]).all() and almost_equal(angle_axis[0] , radians(180) )

#####################

    initial_orientation = angle_axis2quat( np.pi * 0.6 , (0.0,1.0,0.0) )
    initial_translation = (0,0,0)

    sphere_position = [0.0,0.0,0.0]
    ref_points_3D = [
        [0.0,0.5,1.0],
        [0.0,0.0,1.0],
        [0.0,0.0,1.0],
        [0.0,-0.5,1.0]
    ]
    gaze_directions_3D = [
        [ 0.0,0.5,-1.0],
        [ 0.0,0.0,-1.0],
        [ 0.0,0.0,-1.0],
        [ 0.0,-0.5,-1.0]
    ]
    success, orientation, translation = point_line_calibration( sphere_position, ref_points_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    assert almost_equal(angle_axis[1] , [0,1,0,]).all() and almost_equal(angle_axis[0] , radians(180) )

#####################

    initial_orientation = angle_axis2quat( np.pi * 0.25 , (0.0,1.0,0.0) )
    initial_translation = (-10,0,0)

    sphere_position = [0.0,0.0,0.0]
    ref_directions_3D = [
        [0.0,0.0,1.0],
        [0.5,0.0,0.5]
    ]
    gaze_directions_3D = [
        [0.0,0.0,1.0],
        [0.5,0.0,0.5]
    ]
    success, orientation, translation = line_line_calibration( sphere_position, ref_directions_3D, gaze_directions_3D , initial_orientation , initial_translation , fix_translation = True )
    angle_axis =  quat2angle_axis(orientation)
    print success
    print angle_axis
    print np.pi * 0.25
    assert almost_equal(angle_axis[1] , [0,1,0,]).all() and almost_equal(angle_axis[0] , radians(180) )

    print 'Test Ended'


