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
from scipy import optimize as scipy_optimize
from scipy import sparse as scipy_sparse

from . import utils


# BundleAdjustment is a class instead of functions, since passing all the parameters
# would be inefficient. (especially true for _compute_residuals as a callback)
class BundleAdjustment:
    def __init__(self, fix_gaze_targets):
        self._fix_gaze_targets = bool(fix_gaze_targets)
        self._opt_items = None
        self._n_spherical_cameras = None
        self._n_poses_variables = None
        self._gaze_targets_size = None
        self._indices = None
        self._current_values = None
        self._rotation_size = None

        self._row_ind = None
        self._col_ind = None

    @staticmethod
    def _toarray(arr):
        return np.asarray(arr, dtype=np.float64)

    def calculate(self, initial_spherical_cameras, initial_gaze_targets):
        initial_rotation = self._toarray(
            [o.rotation for o in initial_spherical_cameras]
        )
        initial_translation = self._toarray(
            [o.translation for o in initial_spherical_cameras]
        )
        all_observations = self._toarray(
            [o.observations for o in initial_spherical_cameras]
        )
        initial_gaze_targets = self._toarray(initial_gaze_targets)

        opt_rot = [not o.fix_rotation for o in initial_spherical_cameras]
        opt_trans = [not o.fix_translation for o in initial_spherical_cameras]
        self._opt_items = np.array(opt_rot + opt_trans, dtype=bool)
        self._n_poses_variables = 3 * np.sum(self._opt_items)
        self._rotation_size = initial_rotation.size
        self._gaze_targets_size = initial_gaze_targets.size
        self._n_spherical_cameras = len(initial_spherical_cameras)
        self._indices = self._get_indices()
        initial_guess = self._get_initial_guess(
            initial_rotation, initial_translation, initial_gaze_targets
        )
        self._row_ind, self._col_ind = self._get_ind_for_jacobian_matrix()

        result = self._least_squares(initial_guess, all_observations)
        return self._get_final_output(result)

    def _get_indices(self):
        """Get the indices of the parameters for the optimization"""

        to_be_opt = np.repeat(self._opt_items, 3)
        if not self._fix_gaze_targets:
            to_be_opt = np.append(
                to_be_opt, np.ones(self._gaze_targets_size, dtype=bool)
            )
        return np.where(to_be_opt)[0]

    def _get_initial_guess(
        self, initial_rotation, initial_translation, initial_gaze_targets
    ):
        self._current_values = np.append(
            initial_rotation.ravel(), initial_translation.ravel()
        )
        self._current_values = np.append(
            self._current_values, initial_gaze_targets.ravel()
        )
        return self._current_values[self._indices]

    def _get_ind_for_jacobian_matrix(self):
        def get_mat_pose(i):
            mat_pose = np.ones((self._gaze_targets_size, 3), dtype=bool)
            row, col = np.where(mat_pose)
            row += (i % self._n_spherical_cameras) * self._gaze_targets_size
            col += 3 * np.sum(self._opt_items[:i])
            return row, col

        try:
            row_ind, col_ind = np.concatenate(
                [
                    get_mat_pose(i)
                    for i in range(len(self._opt_items))
                    if self._opt_items[i]
                ],
                axis=1,
            )
        except ValueError:
            row_ind, col_ind = np.where([[]])

        if not self._fix_gaze_targets:
            _row = np.repeat(
                np.arange(self._gaze_targets_size).reshape(-1, 3), 3, axis=0
            ).ravel()
            ind_row = [
                _row + self._gaze_targets_size * i
                for i in range(self._n_spherical_cameras)
            ]
            ind_row = np.asarray(ind_row).ravel()
            ind_col = np.tile(
                np.repeat(np.arange(self._gaze_targets_size), 3),
                self._n_spherical_cameras,
            )
            row_ind = np.append(row_ind, ind_row)
            col_ind = np.append(col_ind, ind_col + self._n_poses_variables)

        return row_ind, col_ind

    def _calculate_jacobian_matrix(self, variables, all_observations):
        def get_jac_rot(normals, rotation):
            jacobian = cv2.Rodrigues(rotation)[1].reshape(3, 3, 3)
            return np.einsum("mk,ijk->mji", normals, jacobian)

        def get_jac_trans(translation):
            vectors = gaze_targets - translation
            norms = np.linalg.norm(vectors, axis=1)
            block = -np.einsum("ki,kj->kij", vectors, vectors)
            block /= (norms**3)[:, np.newaxis, np.newaxis]
            ones = np.eye(3)[np.newaxis] / norms[:, np.newaxis, np.newaxis]
            block += ones
            return block

        rotations, translations, gaze_targets = self._decompose_variables(variables)

        data_rot = [
            get_jac_rot(normals, rotation)
            for normals, rotation, opt in zip(
                all_observations,
                rotations,
                self._opt_items[: self._n_spherical_cameras],
            )
            if opt
        ]
        data_rot = self._toarray(data_rot).ravel()
        data_trans = [
            get_jac_trans(translation)
            for translation, opt in zip(
                translations, self._opt_items[self._n_spherical_cameras :]
            )
            if opt
        ]
        data_trans = self._toarray(data_trans).ravel()
        data = np.append(data_rot, data_trans)

        if not self._fix_gaze_targets:
            data_targets = [-get_jac_trans(translation) for translation in translations]
            data_targets = self._toarray(data_targets).ravel()
            data = np.append(data, data_targets)

        n_residuals = self._gaze_targets_size * self._n_spherical_cameras
        n_variables = len(self._indices)
        jacobian_matrix = scipy_sparse.csc_matrix(
            (data, (self._row_ind, self._col_ind)), shape=(n_residuals, n_variables)
        )
        return jacobian_matrix

    def _least_squares(self, initial_guess, all_observations, tol=1e-8, max_nfev=100):
        x_scale = np.ones(self._n_poses_variables)
        if not self._fix_gaze_targets:
            x_scale = np.append(x_scale, np.ones(self._gaze_targets_size) * 500) / 20

        result = scipy_optimize.least_squares(
            fun=self._compute_residuals,
            x0=initial_guess,
            args=(all_observations,),
            jac=self._calculate_jacobian_matrix,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            x_scale=x_scale,
            max_nfev=max_nfev,
            verbose=1,
        )
        return result

    def _compute_residuals(self, variables, all_observations):
        rotations, translations, gaze_targets = self._decompose_variables(variables)

        all_observations_world = self._transform_all_observations_to_world(
            rotations, all_observations
        )
        projected_gaze_targets = self._project_gaze_targets(translations, gaze_targets)
        residuals = all_observations_world - projected_gaze_targets
        return residuals.ravel()

    def _transform_all_observations_to_world(self, rotations, all_observations):
        rotation_matrices = [cv2.Rodrigues(r)[0] for r in rotations]
        all_observations_world = [
            np.dot(matrix, observations.T).T
            for matrix, observations in zip(rotation_matrices, all_observations)
        ]
        return self._toarray(all_observations_world)

    @staticmethod
    def _project_gaze_targets(translations, gaze_targets):
        """Project gaze targets onto the spherical cameras
        (where projection simply means normalization)
        """

        directions = gaze_targets[np.newaxis] - translations[:, np.newaxis]
        norms = np.linalg.norm(directions, axis=2)[:, :, np.newaxis]
        projected_gaze_targets = directions / norms
        return projected_gaze_targets

    def _decompose_variables(self, variables):
        self._current_values[self._indices] = variables
        rotations = self._current_values[: self._rotation_size].reshape(-1, 3)
        translations = self._current_values[
            self._rotation_size : -self._gaze_targets_size
        ].reshape(-1, 3)
        gaze_targets = self._current_values[-self._gaze_targets_size :].reshape(-1, 3)
        return rotations, translations, gaze_targets

    def _get_final_output(self, result):
        residual = result.cost
        rotations, translations, final_gaze_targets = self._decompose_variables(
            result.x
        )
        final_poses = [
            utils.merge_extrinsic(rotation, translation)
            for rotation, translation in zip(rotations, translations)
        ]
        return residual, final_poses, final_gaze_targets
