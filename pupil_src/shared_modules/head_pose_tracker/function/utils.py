"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import functools
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def split_extrinsics(extrinsics):
    extrinsics = np.array(extrinsics, dtype=np.float32)
    assert extrinsics.size == 6
    # extrinsics could be of shape (6,) or (1, 6), so ravel() is needed.
    rotation = extrinsics.ravel()[0:3]
    translation = extrinsics.ravel()[3:6]
    return rotation, translation


def merge_extrinsics(rotation, translation):
    assert rotation.size == 3 and translation.size == 3
    # rotation and translation could be of shape (3,) or (1, 3), so ravel() is needed.
    extrinsics = np.concatenate((rotation.ravel(), translation.ravel()))
    return extrinsics


def to_camera_coordinate(pts_3d_world, rotation, translation):
    pts_3d_cam = [
        np.matmul(cv2.Rodrigues(rotation)[0], p) + translation.ravel()
        for p in pts_3d_world.reshape(-1, 3)
    ]
    pts_3d_cam = np.array(pts_3d_cam)

    return pts_3d_cam


def convert_extrinsic_to_matrix(extrinsics):
    rotation, translation = split_extrinsics(extrinsics)
    extrinsic_matrix = np.eye(4, dtype=np.float32)
    extrinsic_matrix[0:3, 0:3] = cv2.Rodrigues(rotation)[0]
    extrinsic_matrix[0:3, 3] = translation
    return extrinsic_matrix


def convert_matrix_to_extrinsic(extrinsic_matrix):
    rotation = cv2.Rodrigues(extrinsic_matrix[0:3, 0:3])[0]
    translation = extrinsic_matrix[0:3, 3]
    return merge_extrinsics(rotation, translation)


def rod_to_euler(rotation_pose):
    """
    :param rotation_pose: Compact Rodrigues rotation vector, representing
    the rotation axis with its length encoding the angle in radians to rotate
    :return: x,y,z: Orientation in the Pitch, Roll and Yaw axes as Euler angles
    according to 'right hand' convention
    """
    # convert Rodrigues rotation vector to matrix
    rot = cv2.Rodrigues(rotation_pose)[0]

    # rotate 180 degrees in y- and z-axes (corresponds to yaw and roll) to align
    # with right hand rule (relative to the coordinate system of the origin marker
    rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rot = np.matmul(rot, rot_mat)

    # convert to euler angles
    sin_y = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    if not sin_y < 1e-6:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = -np.arctan2(-rot[2, 0], sin_y)
        z = -np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = -np.arctan2(-rot[2, 0], sin_y)
        z = 0

    return np.rad2deg([x, y, z])


def get_camera_pose(camera_extrinsics):
    if camera_extrinsics is None:
        return get_none_camera_extrinsics()

    rotation_ext, translation_ext = split_extrinsics(camera_extrinsics)
    rotation_pose = -rotation_ext
    translation_pose = np.matmul(-cv2.Rodrigues(rotation_ext)[0].T, translation_ext)
    camera_pose = merge_extrinsics(rotation_pose, translation_pose)
    euler_orientation = rod_to_euler(rotation_pose)
    return camera_pose, euler_orientation


def convert_marker_extrinsics_to_points_3d(marker_extrinsics):
    mat = convert_extrinsic_to_matrix(marker_extrinsics)
    marker_transformed_h = np.matmul(mat, get_marker_points_4d_origin().T)
    marker_points_3d = cv2.convertPointsFromHomogeneous(marker_transformed_h.T)
    marker_points_3d.shape = 4, 3

    return marker_points_3d


def get_marker_points_3d_origin():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)


def get_marker_points_4d_origin():
    return np.array(
        [[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32
    )


def get_marker_extrinsics_origin():
    return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)


def get_none_camera_extrinsics():
    return np.full((6,), np.nan)


def find_origin_marker_id(marker_id_to_extrinsics):
    for marker_id, extrinsics in marker_id_to_extrinsics.items():
        if np.allclose(extrinsics, get_marker_extrinsics_origin()):
            return marker_id
    return None


def svdt(A, B):
    """Calculates the transformation between two coordinate systems using SVD.
    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err, where A and B represents the rigid body in different instants
    and err is an aleatory noise (which should be zero for a perfect rigid body).

    Adapted from: https://github.com/demotu/BMC/blob/master/functions/svdt.py
    """

    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3

    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)
    M = np.dot((B - B_centroid).T, (A - A_centroid))
    U, S, Vt = np.linalg.svd(M)

    rotation_matrix = np.dot(
        U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt)
    )

    translation_vector = B_centroid - np.dot(rotation_matrix, A_centroid)

    err = 0
    for i in range(len(A)):
        Bp = np.dot(rotation_matrix, A[i, :]) + translation_vector
        err += np.sum((Bp - B[i, :]) ** 2)
    root_mean_squared_error = np.sqrt(err / A.size)

    return rotation_matrix, translation_vector, root_mean_squared_error


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t1 = time.perf_counter()
        value = func(*args, **kwargs)
        t2 = time.perf_counter()
        run_time = t2 - t1
        if run_time > 1:
            logger.info(f"{func.__name__} took {run_time:.2f} s")
        elif run_time > 1e-3:
            logger.info(f"{func.__name__} took {run_time * 1e3:.2f} ms")
        else:
            logger.info(f"{func.__name__} took {run_time * 1e6:.2f} µs")

        return value

    return wrapper_timer
