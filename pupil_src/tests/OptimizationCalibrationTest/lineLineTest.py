import os, sys, platform
loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
sys.path.append(os.path.join(loc[0], 'pupil_src', 'shared_modules'))

from calibration_routines.optimization_calibration import  line_line_calibration

import numpy as np
import math
from numpy import array, cross, dot, double, hypot, zeros
from math import acos, atan2, cos, pi, sin, radians
from calibration_routines.visualizer_calibration import *
from calibration_routines.calibrate import nearest_intersection_points


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

#from https://github.com/nipy/nibabel with MIT License
def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.
    References
    ----------
    Algorithm from
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    import sys
    if Nq < sys.float_info.epsilon:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ, zZ = y*Y, y*Z, z*Z
    return np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY],
                     [xY+wZ, 1.0-(xX+zZ), yZ-wX],
                     [xZ-wY, yZ+wX, 1.0-(xX+yY)]])




def almost_equal(a, b, accuracy = 10e-6 ):
    return abs(a - b) < accuracy

if __name__ == '__main__':


    from random import uniform

    cam1_center  = (0,0,0)
    cam1_orientation = angle_axis2quat( 0 , (0.0,1.0,0.0) )

    cam2_center  = np.array((500,0,0))
    cam2_orientation = angle_axis2quat( -np.pi/4, (0.0,1.0,0.0) )
    cam2_rotation_matrix = quat2mat(cam2_orientation)
    random_points = [];
    random_points_amount = 10

    x_var = 20
    y_var = 20
    z_var = 10
    z_min = 100
    for i in range(0,random_points_amount):
        random_point = ( uniform(-x_var,x_var) ,  uniform(-y_var,y_var) ,  uniform(z_min,z_min+z_var)  )
        random_points.append(random_point)


    def toEye(p):
        return np.dot(p-cam2_center, cam2_rotation_matrix)

    def toWorld(p):
        return np.dot(p, cam2_rotation_matrix.T) + cam2_center

    cam1_points = [] #cam1 coords
    cam2_points = [] #cam2 coords
    for p in random_points:
        cam1_points.append(p)
        p2 = toEye(p) # to cam2 coordinate system
        cam2_points.append(p2)

    sphere_position = (0,0,0)
    initial_orientation = angle_axis2quat( -np.pi/2, (0.0,1.0,0.0) )
    initial_translation = [c*uniform(1.0,1.0)for c in cam2_center ]
    print 'initial orientation: ' , initial_orientation
    print 'initial translation: ' , initial_translation

    success, orientation, translation = line_line_calibration( sphere_position, cam1_points, cam2_points , initial_orientation , initial_translation , fix_translation = True )

    print orientation
    print quat2angle_axis(orientation)
    print translation
    #assert (orientation== cam2_orientation).all() #and almost_equal(angle_axis[0] , radians(45) )

    #replace with the optimized rotation and translation
    cam2_rotation_matrix = quat2mat(orientation)
    cam2_to_cam1_matrix  = np.matrix(np.eye(4))
    cam2_to_cam1_matrix[:3,:3] = cam2_rotation_matrix
    cam2_translation = np.matrix(translation)
    cam2_translation.shape = (3,1)
    cam2_to_cam1_matrix[:3,3:4] = cam2_translation

    eye = { 'center': (0,0,0), 'radius': 1.0}

    # intersection_points_a = [] #world coords
    # intersection_points_b = [] #cam2 coords
    # for a,b in zip(cam1_points , cam2_points): #world coords , cam2 coords

    #     line_a = (np.array(cam1_center) , np.array(a))
    #     line_b = (np.array(cam2_center) , toWorld(b) ) #convert to world for intersection
    #     ai, bi, _ =  nearest_intersection_points( line_a , line_b ) #world coords
    #     intersection_points_a.append(ai)
    #     intersection_points_b.append( bi )  #cam2 coords , since visualizer expects local coordinates

    visualizer = Calibration_Visualizer(None,None, cam1_points,cam2_to_cam1_matrix ,cam2_points, run_independently = True )
    visualizer.open_window()
    while visualizer.window:
        visualizer.update_window( None, [] , eye)

    print 'Test Ended'


