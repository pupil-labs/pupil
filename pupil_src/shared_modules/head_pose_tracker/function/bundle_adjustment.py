"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import logging

import cv2
import numpy as np
from head_pose_tracker.function import utils
from scipy import optimize as scipy_optimize
from scipy import sparse as scipy_sparse

logger = logging.getLogger(__name__)

BundleAdjustmentResult = collections.namedtuple(
    "BundleAdjustmentResult",
    ["frame_id_to_extrinsics", "marker_id_to_extrinsics", "frame_ids_failed"],
)


# BundleAdjustment is a class instead of functions, since passing all the parameters
# would be inefficient.
# (especially true for _function_compute_residuals as a callback)
class BundleAdjustment:
    def __init__(self, camera_intrinsics, optimize_camera_intrinsics):
        self._camera_intrinsics = camera_intrinsics
        self._optimize_camera_intrinsics = optimize_camera_intrinsics
        self._enough_samples = False
        self._camera_intrinsics_params_size = 4 + camera_intrinsics.D.size

        self._tol = 1e-8
        self._diff_step = 1e-3

        self._marker_ids = []
        self._frame_ids = []

    def calculate(self, initial_guess_result):
        """run bundle adjustment given the initial guess and then check the result of
        markers_3d_model
        """

        self._enough_samples = bool(len(initial_guess_result.key_markers) >= 30)

        self._marker_ids, self._frame_ids = self._set_ids(
            initial_guess_result.frame_id_to_extrinsics,
            initial_guess_result.marker_id_to_extrinsics,
        )
        camera_extrinsics_array, marker_extrinsics_array = self._set_init_array(
            initial_guess_result.frame_id_to_extrinsics,
            initial_guess_result.marker_id_to_extrinsics,
        )
        self._prepare_basic_data(initial_guess_result.key_markers)

        initial_guess_array, bounds, sparsity_matrix = self._prepare_parameters(
            camera_extrinsics_array, marker_extrinsics_array
        )

        try:
            least_sq_result = self._least_squares(
                initial_guess_array, bounds, sparsity_matrix
            )
        except ValueError as err:
            logger.debug(
                f"Value error encountered during optimizing 3d markers model: {err}"
            )
            return None
        else:
            return self._get_result(least_sq_result)

    @staticmethod
    def _set_ids(frame_id_to_extrinsics, marker_id_to_extrinsics):
        origin_marker_id = utils.find_origin_marker_id(marker_id_to_extrinsics)
        marker_ids = [origin_marker_id]
        marker_ids += list(set(marker_id_to_extrinsics.keys()) - {origin_marker_id})
        frame_ids = list(frame_id_to_extrinsics.keys())
        return marker_ids, frame_ids

    def _set_init_array(self, frame_id_to_extrinsics, marker_id_to_extrinsics):
        camera_extrinsics_array = np.array(
            [frame_id_to_extrinsics[frame_id] for frame_id in self._frame_ids]
        )
        marker_extrinsics_array = np.array(
            [marker_id_to_extrinsics[marker_id] for marker_id in self._marker_ids]
        )
        return camera_extrinsics_array, marker_extrinsics_array

    def _prepare_basic_data(self, key_markers):
        self._frame_indices = np.array(
            [self._frame_ids.index(marker.frame_id) for marker in key_markers]
        )
        self._marker_indices = np.array(
            [self._marker_ids.index(marker.marker_id) for marker in key_markers]
        )
        self._markers_points_2d_detected = np.array(
            [marker.verts for marker in key_markers]
        )

    def _prepare_parameters(self, camera_extrinsics_array, marker_extrinsics_array):
        self._camera_extrinsics_shape = camera_extrinsics_array.shape
        self._marker_extrinsics_shape = marker_extrinsics_array.shape

        initial_guess_array = np.vstack(
            (camera_extrinsics_array, marker_extrinsics_array)
        ).ravel()
        if self._optimize_camera_intrinsics and self._enough_samples:
            camera_intrinsics_params = self._load_camera_intrinsics_params(
                self._camera_intrinsics.K, self._camera_intrinsics.D
            )
            initial_guess_array = np.hstack(
                (initial_guess_array, camera_intrinsics_params)
            )

        bounds = self._calculate_bounds()

        sparsity_matrix = self._construct_sparsity_matrix()

        return initial_guess_array, bounds, sparsity_matrix

    def _calculate_bounds(self, eps=np.finfo(np.float64).eps, scale=np.inf):
        """calculate the lower and upper bounds on independent variables
        fix the first marker at the origin of the coordinate system
        """
        camera_extrinsics_lower_bound = np.full(self._camera_extrinsics_shape, -scale)
        camera_extrinsics_upper_bound = np.full(self._camera_extrinsics_shape, scale)

        marker_extrinsics_lower_bound = np.full(self._marker_extrinsics_shape, -scale)
        marker_extrinsics_upper_bound = np.full(self._marker_extrinsics_shape, scale)
        marker_extrinsics_origin = utils.get_marker_extrinsics_origin()
        marker_extrinsics_lower_bound[0] = marker_extrinsics_origin - eps
        marker_extrinsics_upper_bound[0] = marker_extrinsics_origin + eps

        lower_bound = np.vstack(
            (camera_extrinsics_lower_bound, marker_extrinsics_lower_bound)
        ).ravel()
        upper_bound = np.vstack(
            (camera_extrinsics_upper_bound, marker_extrinsics_upper_bound)
        ).ravel()

        if self._optimize_camera_intrinsics and self._enough_samples:
            camera_matrix_lower_bound = np.full(4, 0)
            camera_matrix_upper_bound = np.full(4, 2000)
            dist_coefs_lower_bound = np.full(self._camera_intrinsics.D.size, -1)
            dist_coefs_upper_bound = np.full(self._camera_intrinsics.D.size, 1)
            lower_bound = np.hstack(
                (lower_bound, camera_matrix_lower_bound, dist_coefs_lower_bound)
            )
            upper_bound = np.hstack(
                (upper_bound, camera_matrix_upper_bound, dist_coefs_upper_bound)
            )

        return lower_bound, upper_bound

    def _construct_sparsity_matrix(self):
        """
        Construct the sparsity structure of the Jacobian matrix for finite difference
        estimation. If the Jacobian has only few non-zero elements in each row,
        providing the sparsity structure will greatly speed up the computations. A zero
        entry means that a corresponding element in the Jacobian is identically zero.

        :return: scipy.sparse.lil_matrix, with shape (n_residuals, n_variables), where
        n_residuals = markers_points_2d_detected.size,
        n_variables = camera_extrinsics_array.size + marker_extrinsics_array.size
        """

        n_samples = len(self._frame_indices)

        mat_camera = np.zeros(
            (n_samples, self._camera_extrinsics_shape[0]), dtype=np.uint8
        )
        mat_camera[np.arange(n_samples), self._frame_indices] = 1
        mat_camera = cv2.resize(
            mat_camera,
            (
                np.prod(self._camera_extrinsics_shape),
                self._markers_points_2d_detected.size,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        mat_marker = np.zeros(
            (n_samples, self._marker_extrinsics_shape[0]), dtype=np.uint8
        )
        mat_marker[np.arange(n_samples), self._marker_indices] = 1
        mat_marker = cv2.resize(
            mat_marker,
            (
                np.prod(self._marker_extrinsics_shape),
                self._markers_points_2d_detected.size,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        sparsity_matrix = np.hstack((mat_camera, mat_marker))

        if self._optimize_camera_intrinsics and self._enough_samples:
            mat_camera_intrinsics = np.ones(
                (
                    self._markers_points_2d_detected.size,
                    self._camera_intrinsics_params_size,
                ),
                dtype=int,
            )
            sparsity_matrix = np.hstack((sparsity_matrix, mat_camera_intrinsics))

        sparsity_matrix = scipy_sparse.lil_matrix(sparsity_matrix)
        return sparsity_matrix

    def _least_squares(self, initial_guess_array, bounds, sparsity_matrix):
        result = scipy_optimize.least_squares(
            fun=self._function_compute_residuals,
            x0=initial_guess_array,
            bounds=bounds,
            ftol=self._tol,
            xtol=self._tol,
            gtol=self._tol,
            x_scale="jac",
            loss="soft_l1",
            diff_step=self._diff_step,
            jac_sparsity=sparsity_matrix,
            max_nfev=100,
        )
        return result

    def _get_result(self, least_sq_result):
        camera_extrinsics_array, marker_extrinsics_array = self._get_extrinsics_arrays(
            least_sq_result.x
        )
        frame_indices_failed, marker_indices_failed = self._find_failed_indices(
            least_sq_result.fun
        )

        frame_id_to_extrinsics_opt = {
            self._frame_ids[frame_index]: extrinsics
            for frame_index, extrinsics in enumerate(camera_extrinsics_array)
            if frame_index not in frame_indices_failed
        }
        marker_id_to_extrinsics_opt = {
            self._marker_ids[marker_index]: extrinsics
            for marker_index, extrinsics in enumerate(marker_extrinsics_array)
            if marker_index not in marker_indices_failed or marker_index == 0
        }
        frame_ids_failed = [self._frame_ids[i] for i in frame_indices_failed]

        bundle_adjustment_result = BundleAdjustmentResult(
            frame_id_to_extrinsics_opt, marker_id_to_extrinsics_opt, frame_ids_failed
        )
        return bundle_adjustment_result

    def _function_compute_residuals(self, variables):
        """Function which computes the vector of residuals,
        i.e., the minimization proceeds with respect to params
        """
        camera_extrinsics_array, marker_extrinsics_array = self._get_extrinsics_arrays(
            variables
        )
        if self._optimize_camera_intrinsics and self._enough_samples:
            self._unload_camera_intrinsics_params(
                variables[-self._camera_intrinsics_params_size :]
            )

        markers_points_2d_projected = self._project_markers(
            camera_extrinsics_array, marker_extrinsics_array
        )
        residuals = markers_points_2d_projected - self._markers_points_2d_detected
        return residuals.ravel()

    def _get_extrinsics_arrays(self, variables):
        """reshape 1-dimensional vector into the original shape of
        camera_extrinsics_array and marker_extrinsics_array
        """

        variables = variables[
            : np.prod(self._camera_extrinsics_shape)
            + np.prod(self._marker_extrinsics_shape)
        ].copy()

        camera_extrinsics_array = variables[: np.prod(self._camera_extrinsics_shape)]
        camera_extrinsics_array.shape = self._camera_extrinsics_shape

        marker_extrinsics_array = variables[-np.prod(self._marker_extrinsics_shape) :]
        marker_extrinsics_array.shape = self._marker_extrinsics_shape

        return camera_extrinsics_array, marker_extrinsics_array

    def _project_markers(self, camera_extrinsics_array, marker_extrinsics_array):
        markers_points_3d = np.array(
            [
                utils.convert_marker_extrinsics_to_points_3d(extrinsics)
                for extrinsics in marker_extrinsics_array
            ]
        )
        markers_points_2d_projected = np.array(
            [
                self._camera_intrinsics.projectPoints(points, cam[0:3], cam[3:6])
                for cam, points in zip(
                    camera_extrinsics_array[self._frame_indices],
                    markers_points_3d[self._marker_indices],
                )
            ]
        )
        return markers_points_2d_projected

    def _find_failed_indices(self, residuals, thres_frame=8, thres_marker=8):
        """find out those frame_indices and marker_indices which cause large
        reprojection errors
        """

        residuals.shape = -1, 4, 2
        reprojection_errors = np.linalg.norm(residuals, axis=2).sum(axis=1)

        frame_indices_failed = [
            frame_indice
            for frame_indice in set(self._frame_indices)
            if np.min(reprojection_errors[self._frame_indices == frame_indice])
            > thres_frame
        ]
        marker_indices_failed = [
            marker_indice
            for marker_indice in set(self._marker_indices)
            if np.min(reprojection_errors[self._marker_indices == marker_indice])
            > thres_marker
        ]
        return frame_indices_failed, marker_indices_failed

    def _load_camera_intrinsics_params(self, camera_matrix, dist_coefs):
        camera_intrinsics_params = np.zeros(self._camera_intrinsics_params_size)

        camera_intrinsics_params[0] = camera_matrix[0, 0]  # fx
        camera_intrinsics_params[1] = camera_matrix[1, 1]  # fy
        camera_intrinsics_params[2] = camera_matrix[0, 2]  # cx
        camera_intrinsics_params[3] = camera_matrix[1, 2]  # cy

        camera_intrinsics_params[4:] = dist_coefs.ravel()

        return camera_intrinsics_params

    def _unload_camera_intrinsics_params(self, camera_intrinsics_params):
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = camera_intrinsics_params[0]  # fx
        camera_matrix[1, 1] = camera_intrinsics_params[1]  # fy
        camera_matrix[0, 2] = camera_intrinsics_params[2]  # cx
        camera_matrix[1, 2] = camera_intrinsics_params[3]  # cy

        dist_coefs = camera_intrinsics_params[4:]

        self._camera_intrinsics.update_camera_matrix(camera_matrix)
        self._camera_intrinsics.update_dist_coefs(dist_coefs)
