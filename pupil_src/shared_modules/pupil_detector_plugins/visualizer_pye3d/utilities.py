"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import numpy as np


def cart2sph(x):
    phi = np.arctan2(x[2], x[0])
    theta = np.arccos(x[1] / np.linalg.norm(x))

    return phi, theta  # Todo: This seems to be opposite to the pupil code


def sph2cart(phi, theta):
    result = np.empty(3)

    result[0] = np.sin(theta) * np.cos(phi)
    result[1] = np.cos(theta)
    result[2] = np.sin(theta) * np.sin(phi)

    return result


def normalize(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis)


def enclosed_angle(v1, v2, unit="deg", axis=-1):
    v1 = normalize(v1, axis=axis)
    v2 = normalize(v2, axis=axis)

    alpha = np.arccos(np.clip(np.dot(v1.T, v2), -1, 1))

    if unit == "deg":
        return 180.0 / np.pi * alpha
    else:
        return alpha


def make_homogeneous_vector(v):
    return np.hstack((v, [0.0]))


def make_homogeneous_point(p):
    return np.hstack((p, [1.0]))


def transform_as_homogeneous_point(p, trafo):
    p = make_homogeneous_point(p)
    return (trafo @ p)[:3]


def transform_as_homogeneous_vector(v, trafo):
    v = make_homogeneous_vector(v)
    return (trafo @ v)[:3]


def rotate_v1_on_v2(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    cos_angle = np.dot(v1, v2)

    if not np.allclose(np.abs(cos_angle), 1):
        u = np.cross(v1, v2)
        s = np.linalg.norm(u)
        c = np.dot(v1, v2)

        I = np.eye(3)
        ux = np.asarray([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

        R = I + ux + np.dot(ux, ux) * (1 - c) / s**2

    elif np.allclose(cos_angle, 1):
        R = np.eye(3)

    elif np.allclose(cos_angle, -1):
        R = -np.eye(3)

    return R
