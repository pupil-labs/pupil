"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import numpy as np


def transform_points_by_extrinsic(points_3d_cam1, extrinsic_cam2_cam1):
    """
    Transform 3d points from cam1 coordinate to cam2 coordinate

    :param points_3d_cam1: 3d points in cam1 coordinate, shape: (N x 3)
    :param extrinsic_cam2_cam1: extrinsic of cam2 in cam1 coordinate, shape: (6,)
    :return: 3d points in cam2 coordinate, shape: (N x 3)
    """

    rotation_cam2_cam1, translation_cam2_cam1 = split_extrinsic(extrinsic_cam2_cam1)
    points_3d_cam1 = np.asarray(points_3d_cam1, dtype=np.float64)
    points_3d_cam1.shape = -1, 3
    rotation_matrix_cam2_cam1 = cv2.Rodrigues(rotation_cam2_cam1)[0]
    points_3d_cam2 = (
        np.dot(rotation_matrix_cam2_cam1, points_3d_cam1.T).T + translation_cam2_cam1
    )
    return points_3d_cam2


def transform_points_by_pose(points_3d_cam1, pose_cam2_cam1):
    """
    Transform 3d points from cam1 coordinate to cam2 coordinate

    :param points_3d_cam1: 3d points in cam1 coordinate, shape: (N x 3)
    :param pose_cam2_cam1: camera pose of cam2 in cam1 coordinate, shape: (6,)
    :return: 3d points in cam2 coordinate, shape: (N x 3)
    """

    rotation_cam2_cam1, translation_cam2_cam1 = split_extrinsic(pose_cam2_cam1)
    points_3d_cam1 = np.asarray(points_3d_cam1, dtype=np.float64)
    points_3d_cam1.shape = -1, 3

    rotation_matrix_cam2_cam1 = cv2.Rodrigues(rotation_cam2_cam1)[0]
    rotation_matrix_cam1_cam2 = rotation_matrix_cam2_cam1.T
    translation_cam1_cam2 = np.dot(-rotation_matrix_cam1_cam2, translation_cam2_cam1)
    points_3d_cam2 = (
        np.dot(rotation_matrix_cam1_cam2, points_3d_cam1.T).T + translation_cam1_cam2
    )
    return points_3d_cam2


def inverse_extrinsic(extrinsic):
    rotation_ext, translation_ext = split_extrinsic(extrinsic)
    rotation_inv = -rotation_ext
    translation_inv = np.dot(-cv2.Rodrigues(rotation_inv)[0], translation_ext)
    return merge_extrinsic(rotation_inv, translation_inv)


def split_extrinsic(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    assert extrinsic.size == 6
    rotation = extrinsic.ravel()[0:3]
    translation = extrinsic.ravel()[3:6]
    return rotation, translation


def merge_extrinsic(rotation, translation):
    assert rotation.size == 3 and translation.size == 3
    extrinsic = np.concatenate((rotation.ravel(), translation.ravel()))
    return extrinsic


def find_rigid_transform(A, B):
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
    return rotation_matrix, translation_vector


def get_initial_eye_camera_rotation(pupil_normals, gaze_targets):
    initial_rotation_matrix, _ = find_rigid_transform(pupil_normals, gaze_targets)
    initial_rotation = cv2.Rodrigues(initial_rotation_matrix)[0].ravel()
    return initial_rotation


def get_eye_cam_pose_in_world(eye_pose, sphere_pos):
    """
    :param eye_pose: eye pose in world coordinates
    :param sphere_pos: eye ball center in eye cam coordinates
    :return: the eye cam pose in world coordinates
    """

    eye_cam_position_in_eye = -np.asarray(sphere_pos)
    world_extrinsic = eye_pose
    eye_cam_position_in_world = transform_points_by_extrinsic(
        eye_cam_position_in_eye, world_extrinsic
    )

    rotation, translation = split_extrinsic(eye_pose)
    eye_cam_rotation_in_world = cv2.Rodrigues(rotation)[0]

    eye_cam_pose_in_world = np.eye(4)
    eye_cam_pose_in_world[0:3, 0:3] = eye_cam_rotation_in_world
    eye_cam_pose_in_world[0:3, 3] = eye_cam_position_in_world
    return eye_cam_pose_in_world


def calculate_nearest_linepoints_to_points(ref_points, lines):
    p1, p2 = lines
    direction = p2 - p1
    denom = np.linalg.norm(direction, axis=1)
    denom[denom == 0] = 1
    delta = np.diag(np.dot(ref_points - p1, direction.T)) / (denom * denom)
    nearest_linepoints = p1 + direction * delta[:, np.newaxis]
    return nearest_linepoints


def calculate_nearest_points_to_targets(
    all_observations, poses_in_world, gaze_targets_in_world
):
    all_nearest_points = []
    for observations, pose in zip(all_observations, poses_in_world):
        lines_start = transform_points_by_extrinsic(np.zeros(3), pose)
        lines_end = transform_points_by_extrinsic(observations, pose)
        nearest_points = calculate_nearest_linepoints_to_points(
            gaze_targets_in_world, (lines_start, lines_end)
        )
        all_nearest_points.append(nearest_points)

    return all_nearest_points


def _clamp_norm_point(pos):
    """realistic numbers for norm pos should be in this range.
    Grossly bigger or smaller numbers are results bad exrapolation
    and can cause overflow erorr when denormalized and cast as int32.
    """
    return min(100.0, max(-100.0, pos[0])), min(100.0, max(-100.0, pos[1]))
