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

from . import bundle_adjustment, utils

# Fixed eyeball positions are assumed for all users
eye0_hardcoded_translation = np.array([20, 15, -20])
eye1_hardcoded_translation = np.array([-40, 15, -20])

residual_threshold = 1e3


class SphericalCamera:
    def __init__(
        self, observations, rotation, translation, fix_rotation, fix_translation
    ):
        self.observations = observations
        self.rotation = rotation
        self.translation = translation
        self.fix_rotation = bool(fix_rotation)
        self.fix_translation = bool(fix_translation)


def calibrate_binocular(
    unprojected_ref_points,
    pupil0_normals,
    pupil1_normals,
    initial_depth,
    initial_translation0,
    initial_translation1,
):
    """Determine the poses of the eyes and 3d gaze points by solving a specific
    least-squares minimization

    :param unprojected_ref_points: the unprojection of the observed 2d reference points
    at unit distance in world camera coordinates
    :param pupil0_normals: eye0's pupil normals in eye0 camera coordinates
    :param pupil1_normals: eye1's pupil normals in eye1 camera coordinates
    :param initial_depth: initial guess of the depth of the gaze targets
    :return: optimized poses and 3d gaze targets in world camera coordinates
    """
    # Binocular calibration strategy:
    # Take world cam as the origin and express everything in world cam coordinates.
    # Minimize reprojection-type errors by moving the 3d gaze targets and
    # adjusting the orientation of the eyes while fixing their positions.

    # Find initial guess for the poses in world coordinates
    initial_rotation0 = utils.get_initial_eye_camera_rotation(
        pupil0_normals, unprojected_ref_points
    )
    initial_rotation1 = utils.get_initial_eye_camera_rotation(
        pupil1_normals, unprojected_ref_points
    )

    # world cam and eyes are viewed as spherical cameras of unit radius
    world = SphericalCamera(
        observations=unprojected_ref_points,
        rotation=np.zeros(3),
        translation=np.zeros(3),
        fix_rotation=True,
        fix_translation=True,
    )
    eye0 = SphericalCamera(
        observations=pupil0_normals,
        rotation=initial_rotation0,
        translation=initial_translation0,
        fix_rotation=False,
        fix_translation=True,
    )
    eye1 = SphericalCamera(
        observations=pupil1_normals,
        rotation=initial_rotation1,
        translation=initial_translation1,
        fix_rotation=False,
        fix_translation=True,
    )

    initial_spherical_cameras = world, eye0, eye1
    initial_gaze_targets = unprojected_ref_points * initial_depth

    ba = bundle_adjustment.BundleAdjustment(fix_gaze_targets=False)
    residual, poses_in_world, gaze_targets_in_world = ba.calculate(
        initial_spherical_cameras, initial_gaze_targets
    )

    success = residual < residual_threshold
    return success, poses_in_world, gaze_targets_in_world


def calibrate_monocular(
    unprojected_ref_points, pupil_normals, pupil_id, initial_depth, initial_translation
):
    """Determine the poses of the eyes and 3d gaze points by solving a specific
    least-squares minimization

    :param unprojected_ref_points: the unprojection of the observed 2d reference points
    at unit distance in world camera coordinates
    :param pupil_normals: eye's pupil normals in eye camera coordinates
    :param pupil_id: pupil id (0 or 1)
    :param initial_depth: initial guess of the depth of the gaze targets
    :return: optimized poses and 3d gaze targets in world camera coordinates
    """
    # Monocular calibration strategy:
    # Take eye as the origin and express everything in eye coordinates.
    # Minimize reprojection-type errors by moving world cam
    # while fixing the 3d gaze targets.

    # Find initial guess for the poses in eye coordinates
    initial_rotation_matrix, _ = utils.find_rigid_transform(
        unprojected_ref_points, pupil_normals
    )
    initial_rotation = cv2.Rodrigues(initial_rotation_matrix)[0].ravel()
    initial_translation = -np.dot(initial_rotation_matrix, initial_translation)

    # world cam and eye are viewed as spherical cameras of unit radius
    world = SphericalCamera(
        observations=unprojected_ref_points,
        rotation=initial_rotation,
        translation=initial_translation,
        fix_rotation=False,
        fix_translation=False,
    )
    eye = SphericalCamera(
        observations=pupil_normals,
        rotation=np.zeros(3),
        translation=np.zeros(3),
        fix_rotation=True,
        fix_translation=True,
    )

    initial_spherical_cameras = world, eye
    initial_gaze_targets = pupil_normals * initial_depth

    ba = bundle_adjustment.BundleAdjustment(fix_gaze_targets=True)
    residual, poses_in_eye, gaze_targets_in_eye = ba.calculate(
        initial_spherical_cameras, initial_gaze_targets
    )

    world_pose_in_eye, eye_pose_in_eye = poses_in_eye

    # Transform everything from eye coordinates to world coordinates
    eye_pose_in_world = utils.inverse_extrinsic(world_pose_in_eye)
    poses_in_world = [np.zeros(6), eye_pose_in_world]
    gaze_targets_in_world = utils.transform_points_by_pose(
        gaze_targets_in_eye, world_pose_in_eye
    )

    success = residual < residual_threshold
    return success, poses_in_world, gaze_targets_in_world


def calibrate_hmd(
    ref_points_3d, pupil0_normals, pupil1_normals, eye_translations, y_flip_factor=-1.0
):
    """Determine the poses of the eyes and 3d gaze points by solving a specific
    least-squares minimization

    :param ref_points_3d: the observed 3d reference points in world camera coordinates
    :param pupil0_normals: eye0's pupil normals in eye0 camera coordinates
    :param pupil1_normals: eye1's pupil normals in eye1 camera coordinates
    :param eye_translations: eyeballs position in world coordinates
    :return: optimized poses and 3d gaze targets in world camera coordinates
    """
    # HMD calibration strategy:
    # Take world cam as the origin and express everything in world cam coordinates.
    # Minimize reprojection-type errors by adjusting the orientation of the eyes
    # while fixing their positions and the 3d gaze targets.

    initial_translation0, initial_translation1 = np.asarray(eye_translations)

    smallest_residual = 1000
    scales = list(np.linspace(0.7, 10, 5))  # TODO: change back to 50
    for s in scales:
        scaled_ref_points_3d = ref_points_3d * (1, y_flip_factor, s)

        # Find initial guess for the poses in eye coordinates
        initial_rotation0 = utils.get_initial_eye_camera_rotation(
            pupil0_normals, scaled_ref_points_3d
        )
        initial_rotation1 = utils.get_initial_eye_camera_rotation(
            pupil1_normals, scaled_ref_points_3d
        )

        eye0 = SphericalCamera(
            observations=pupil0_normals,
            rotation=initial_rotation0,
            translation=initial_translation0,
            fix_rotation=False,
            fix_translation=True,
        )
        eye1 = SphericalCamera(
            observations=pupil1_normals,
            rotation=initial_rotation1,
            translation=initial_translation1,
            fix_rotation=False,
            fix_translation=True,
        )

        initial_spherical_cameras = eye0, eye1
        initial_gaze_targets = scaled_ref_points_3d

        ba = bundle_adjustment.BundleAdjustment(fix_gaze_targets=True)
        residual, poses_in_world, gaze_targets_in_world = ba.calculate(
            initial_spherical_cameras, initial_gaze_targets
        )
        if residual <= smallest_residual:
            smallest_residual = residual
            scales[-1] = s

    success = residual < residual_threshold
    return success, poses_in_world, gaze_targets_in_world
