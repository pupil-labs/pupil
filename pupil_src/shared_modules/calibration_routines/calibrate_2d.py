"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.optimize

# logging
import logging

logger = logging.getLogger(__name__)


class PolynomialMonocular:
    def __init__(self, params, dof=2, degree=2):

        self.dof = dof
        self.degree = degree
        self.func = [poly.polyval, poly.polyval2d, poly.polyval3d][self.dof - 1]
        self.params = np.reshape(params, self.dof * [degree + 1])

    def __call__(self, x):

        return self.func(*x, self.params)


class PolynomialBinocular:
    def __init__(self, params, dof=2, degree=2):

        self.dof = dof
        self.degree = degree

        self.mapping_eye_0 = PolynomialMonocular(
            params[: len(params) // 2], self.dof, self.degree
        )
        self.mapping_eye_1 = PolynomialMonocular(
            params[len(params) // 2 :], self.dof, self.degree
        )

    def __call__(self, x):

        x0 = x[: self.dof]
        x1 = x[self.dof :]

        return 1 / 2 * (self.mapping_eye_0(x0) + self.mapping_eye_1(x1))


class MappingFunction2D:
    def __init__(self, params_x, params_y, dof=2, degree=2, binocular=False):

        if binocular:
            polynomial_type = PolynomialBinocular
        else:
            polynomial_type = PolynomialMonocular

        self.map_to_world_cam_x = polynomial_type(params_x, dof, degree)
        self.map_to_world_cam_y = polynomial_type(params_y, dof, degree)

    def __call__(self, pt):

        return self.map_to_world_cam_x(pt), self.map_to_world_cam_y(pt)


def convert_to_linear_index(multi_idx, size_per_dim):
    dims = len(multi_idx)
    linear_idx = 0
    for k, idx in enumerate(multi_idx):
        linear_idx += idx * (size_per_dim ** (dims - 1 - k))
    return linear_idx


def extend_params(
    params, degree, ignored_terms=[], fill_value=0, binocular=False
):

    params = list(params)

    if binocular:
        n_params = len(params)
        augmented_params_0 = extend_params(
            list(params[: n_params // 2]), degree, ignored_terms
        )
        augmented_params_1 = extend_params(
            list(params[n_params // 2 :]), degree, ignored_terms
        )
        augmented_params = np.hstack((augmented_params_0, augmented_params_1))
    else:
        ignored_terms.sort(
            key=lambda multi_idx: convert_to_linear_index(multi_idx, degree + 1)
        )
        for entry in ignored_terms:
            params.insert(convert_to_linear_index(entry, degree + 1), fill_value)
        augmented_params = np.asarray(params)

    return augmented_params


def cauchy_loss(residuals, loss_scale):
    return loss_scale ** 2 * np.log(1 + residuals / loss_scale ** 2)


def calibration_residuals(
    params,
    pupil_positions,
    targets,
    binocular,
    dof=2,
    degree=2,
    ignored_terms=(),
    regularization=0.0,
    loss_scale=-1,
):

    augmented_params = extend_params(params, degree, ignored_terms, 0, binocular)

    if binocular:
        polynomial_ = PolynomialBinocular(augmented_params, dof, degree)
    else:
        polynomial_ = PolynomialMonocular(augmented_params, dof, degree)

    squared_residuals = (
        polynomial_([column for column in pupil_positions.T]) - targets
    ) ** 2

    if loss_scale > 0:
        cost = 1 / 2 * np.sum(cauchy_loss(squared_residuals, loss_scale))
    else:
        cost = 1 / 2 * np.sum(squared_residuals)

    cost += regularization * np.sum(params ** 2)

    return cost


def fit_mapping_polynomials(
    calibration_data,
    binocular,
    degree=2,
    ignored_terms=(),
    regularization=0.0,
    loss_scale=-1,
):

    calibration_data = np.asarray(calibration_data)

    if binocular:
        dof_multiplier = 2
    else:
        dof_multiplier = 1

    dof = (calibration_data.shape[1] - 2) // dof_multiplier
    n_params = dof_multiplier * ((degree + 1) ** dof - len(ignored_terms))

    success_x, params_x = optimize_parameters(
        n_params,
        calibration_data[:, :-2],
        calibration_data[:, -2],
        binocular,
        dof=dof,
        degree=degree,
        ignored_terms=ignored_terms,
        regularization=regularization,
        loss_scale=loss_scale,
    )

    success_y, params_y = optimize_parameters(
        n_params,
        calibration_data[:, :-2],
        calibration_data[:, -1],
        binocular,
        dof=dof,
        degree=degree,
        ignored_terms=ignored_terms,
        regularization=regularization,
        loss_scale=loss_scale,
    )

    return (
        (success_x and success_y),
        (params_x, params_y, dof, degree, binocular),
    )


def optimize_parameters(
    n_params,
    pupil_positions,
    targets,
    binocular,
    dof=2,
    degree=2,
    ignored_terms=(),
    regularization=0.0,
    loss_scale=-1,
):

    initial_guess = np.zeros(n_params)

    if (0, 0) not in ignored_terms:
        initial_guess[0] = np.mean(targets)
    if binocular:
        initial_guess[n_params // 2] = initial_guess[0]

    fit_result = scipy.optimize.minimize(
        calibration_residuals,
        initial_guess,
        args=(
            pupil_positions,
            targets,
            binocular,
            dof,
            degree,
            ignored_terms,
            regularization,
            loss_scale,
        ),
    )

    final_params = list(
        extend_params(fit_result.x.tolist(), degree, ignored_terms, 0, binocular)
    )

    return fit_result.success, final_params
