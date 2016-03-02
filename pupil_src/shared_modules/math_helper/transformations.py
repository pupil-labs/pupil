# From http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import sys
import numpy as np

def quat2mat(quaternion):

    """Return homogeneous rotation matrix from quaternion.
    """
    q = quaternion
    n = np.dot(q, q)
    if n < np.finfo(np.float64).eps:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


#  From https://github.com/nipy/nibabel/
#  The MIT License

#  Copyright (c) 2009-2014 Matthew Brett <matthew.brett@gmail.com>
#  Copyright (c) 2010-2013 Stephan Gerhard <git@unidesign.ch>
#  Copyright (c) 2006-2014 Michael Hanke <michael.hanke@gmail.com>
#  Copyright (c) 2011 Christian Haselgrove <christian.haselgrove@umassmed.edu>
#  Copyright (c) 2010-2011 Jarrod Millman <jarrod.millman@gmail.com>
#  Copyright (c) 2011-2014 Yaroslav Halchenko <debian@onerussian.com>

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

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
        [Qxx - Qyy - Qzz, 0, 0, 0],
        [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
        [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]
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
    w, x, y, z = unit_vector(quat)
    vec = np.asarray([x, y, z])
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(vec.dtype).eps * 3
        except ValueError:  # integer type
            identity_thresh = np.finfo(np.float64).eps * 3
    n = np.sqrt(x*x + y*y + z*z)
    if n < identity_thresh:
        # if vec is nearly 0,0,0, this is an identity rotation
        return 0.0, np.array([1.0, 0, 0])
    return 2 * np.arccos(w), vec / n

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
        vector = vector / np.sqrt(np.dot(vector, vector))
    t2 = theta / 2.0
    st2 = np.sin(t2)
    return np.concatenate(([np.cos(t2)],
                           vector * st2))

