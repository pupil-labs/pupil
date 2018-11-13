"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

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
        self.degree_per_dof = np.asarray(self.dof * [degree], dtype=np.int)

        np_polynomial_functions = [poly.polyval, poly.polyval2d, poly.polyval3d]
        self.func = np_polynomial_functions[self.dof - 1]
        self.params = np.reshape(params, self.degree_per_dof + 1)

    def __call__(self, x):

        return self.func(*x, self.params)


class PolynomialBinocular:
    def __init__(self, params, dof=2, degree=2):

        self.dof = dof
        self.degree = degree

        self.mapping_eye_0 = PolynomialMonocular(params[: len(params) // 2], self.dof, self.degree)
        self.mapping_eye_1 = PolynomialMonocular(params[len(params) // 2 :], self.dof, self.degree)

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


def cauchy_loss(residual, scale):
    return scale ** 2 * np.log(1 + residual / scale ** 2)


def calibration_objective(
    params, x, targets, polynomial_type, dof=2, degree=2, regularization=0.0, loss_scale=0.2
):

    polynomial_ = polynomial_type(params, dof, degree)
    squared_residuals = (polynomial_([column for column in x.T]) - targets) ** 2

    if loss_scale > 0:
        cost = 1 / 2 * np.sum(cauchy_loss(squared_residuals, loss_scale))
    else:
        cost = 1 / 2 * np.sum(squared_residuals)

    cost += regularization * np.sum(params ** 2)

    return cost


def calibrate_polynomial(
    cal_pt_cloud, binocular=False, degree=2, regularization=0.0, loss_scale=0.2
):

    cal_pt_cloud = np.asarray(cal_pt_cloud)

    if binocular:
        dof_multiplier = 2
        polynomial_type = PolynomialBinocular
    else:
        dof_multiplier = 1
        polynomial_type = PolynomialMonocular

    dof = (cal_pt_cloud.shape[1] - 2) // dof_multiplier
    n_params = dof_multiplier * (degree + 1) ** dof
    initial_guess = np.zeros(n_params)

    fit_x = scipy.optimize.minimize(
        calibration_objective,
        initial_guess,
        args=(
            cal_pt_cloud[:, :-2],
            cal_pt_cloud[:, -2],
            polynomial_type,
            dof,
            degree,
            regularization,
            loss_scale,
        ),
    )
    fit_y = scipy.optimize.minimize(
        calibration_objective,
        initial_guess,
        args=(
            cal_pt_cloud[:, :-2],
            cal_pt_cloud[:, -1],
            polynomial_type,
            dof,
            degree,
            regularization,
            loss_scale,
        ),
    )
    return (
        (fit_x.success and fit_y.success),
        (fit_x.x.tolist(), fit_y.x.tolist(), dof, degree, binocular),
    )


# This function will be simpler once we do not have to support old calibration files.
def build_mapping_function(*args):

    if len(args) == 3:
        return make_map_function_legacy(*args)
    else:
        return MappingFunction2D(*args)


# This function will be gone, once we switch to the new version. For the moment it is needed to work
# with the Pupil standard calibration parameters.
def make_map_function_legacy(cx, cy, n):
    if n == 3:

        def fn(pt):
            X, Y = pt
            x2 = cx[0] * X + cx[1] * Y + cx[2]
            y2 = cy[0] * X + cy[1] * Y + cy[2]
            return x2, y2

    elif n == 5:

        def fn(pt_0, pt_1):
            #        X0        Y0        X1        Y1        Ones
            X0, Y0 = pt_0
            X1, Y1 = pt_1
            x2 = cx[0] * X0 + cx[1] * Y0 + cx[2] * X1 + cx[3] * Y1 + cx[4]
            y2 = cy[0] * X0 + cy[1] * Y0 + cy[2] * X1 + cy[3] * Y1 + cy[4]
            return x2, y2

    elif n == 7:

        def fn(pt):
            X, Y = pt
            x2 = (
                cx[0] * X
                + cx[1] * Y
                + cx[2] * X * X
                + cx[3] * Y * Y
                + cx[4] * X * Y
                + cx[5] * Y * Y * X * X
                + cx[6]
            )
            y2 = (
                cy[0] * X
                + cy[1] * Y
                + cy[2] * X * X
                + cy[3] * Y * Y
                + cy[4] * X * Y
                + cy[5] * Y * Y * X * X
                + cy[6]
            )
            return x2, y2

    elif n == 9:

        def fn(pt):
            #          X         Y         XX         YY         XY         XXYY         XXY         YYX         Ones
            X, Y = pt
            x2 = (
                cx[0] * X
                + cx[1] * Y
                + cx[2] * X * X
                + cx[3] * Y * Y
                + cx[4] * X * Y
                + cx[5] * Y * Y * X * X
                + cx[6] * Y * X * X
                + cx[7] * Y * Y * X
                + cx[8]
            )
            y2 = (
                cy[0] * X
                + cy[1] * Y
                + cy[2] * X * X
                + cy[3] * Y * Y
                + cy[4] * X * Y
                + cy[5] * Y * Y * X * X
                + cy[6] * Y * X * X
                + cy[7] * Y * Y * X
                + cy[8]
            )
            return x2, y2

    elif n == 13:

        def fn(pt_0, pt_1):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1        Ones
            X0, Y0 = pt_0
            X1, Y1 = pt_1
            x2 = (
                cx[0] * X0
                + cx[1] * Y0
                + cx[2] * X1
                + cx[3] * Y1
                + cx[4] * X0 * X0
                + cx[5] * Y0 * Y0
                + cx[6] * X0 * Y0
                + cx[7] * X0 * X0 * Y0 * Y0
                + cx[8] * X1 * X1
                + cx[9] * Y1 * Y1
                + cx[10] * X1 * Y1
                + cx[11] * X1 * X1 * Y1 * Y1
                + cx[12]
            )
            y2 = (
                cy[0] * X0
                + cy[1] * Y0
                + cy[2] * X1
                + cy[3] * Y1
                + cy[4] * X0 * X0
                + cy[5] * Y0 * Y0
                + cy[6] * X0 * Y0
                + cy[7] * X0 * X0 * Y0 * Y0
                + cy[8] * X1 * X1
                + cy[9] * Y1 * Y1
                + cy[10] * X1 * Y1
                + cy[11] * X1 * X1 * Y1 * Y1
                + cy[12]
            )
            return x2, y2

    elif n == 17:

        def fn(pt_0, pt_1):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1            X0X1            X0Y1            Y0X1        Y0Y1           Ones
            X0, Y0 = pt_0
            X1, Y1 = pt_1
            x2 = (
                cx[0] * X0
                + cx[1] * Y0
                + cx[2] * X1
                + cx[3] * Y1
                + cx[4] * X0 * X0
                + cx[5] * Y0 * Y0
                + cx[6] * X0 * Y0
                + cx[7] * X0 * X0 * Y0 * Y0
                + cx[8] * X1 * X1
                + cx[9] * Y1 * Y1
                + cx[10] * X1 * Y1
                + cx[11] * X1 * X1 * Y1 * Y1
                + cx[12] * X0 * X1
                + cx[13] * X0 * Y1
                + cx[14] * Y0 * X1
                + cx[15] * Y0 * Y1
                + cx[16]
            )
            y2 = (
                cy[0] * X0
                + cy[1] * Y0
                + cy[2] * X1
                + cy[3] * Y1
                + cy[4] * X0 * X0
                + cy[5] * Y0 * Y0
                + cy[6] * X0 * Y0
                + cy[7] * X0 * X0 * Y0 * Y0
                + cy[8] * X1 * X1
                + cy[9] * Y1 * Y1
                + cy[10] * X1 * Y1
                + cy[11] * X1 * X1 * Y1 * Y1
                + cy[12] * X0 * X1
                + cy[13] * X0 * Y1
                + cy[14] * Y0 * X1
                + cy[15] * Y0 * Y1
                + cy[16]
            )
            return x2, y2

    else:
        raise Exception("ERROR: unsopported number of coefficiants.")

    return fn