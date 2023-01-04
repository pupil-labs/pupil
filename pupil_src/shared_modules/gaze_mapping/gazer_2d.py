"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import typing as T

import numpy as np
from gaze_mapping.gazer_base import GazerBase, Model, NotEnoughDataError
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


_REFERENCE_FEATURE_COUNT = 2

_MONOCULAR_FEATURE_COUNT = 2
_MONOCULAR_PUPIL_NORM_POS = slice(0, 2)

_BINOCULAR_FEATURE_COUNT = 4
_BINOCULAR_PUPIL_NORM_POS = slice(2, 4)


class Model2D(Model):
    def __init__(self, screen_size=(1, 1), outlier_threshold_pixel=70):
        self.screen_size = screen_size
        self.outlier_threshold_pixel = outlier_threshold_pixel
        self._is_fitted = False
        self._regressor = LinearRegression(fit_intercept=True)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def set_params(self, **params):
        if params == {}:
            return
        for key, value in params.items():
            setattr(self._regressor, key, np.asarray(value))
        self._is_fitted = True

    def get_params(self):
        has_coef = hasattr(self._regressor, "coef_")
        has_intercept = hasattr(self._regressor, "intercept_")
        if not has_coef or not has_intercept:
            return {}
        return {
            "coef_": self._regressor.coef_.tolist(),
            "intercept_": self._regressor.intercept_.tolist(),
        }

    def fit(self, X, Y, outlier_removal_iterations=1):
        assert X.shape[0] == Y.shape[0], "Required shape: (n_samples, n_features)"
        self._validate_feature_dimensionality(X)
        self._validate_reference_dimensionality(Y)

        if X.shape[0] == 0:
            raise NotEnoughDataError

        polynomial_features = self._polynomial_features(X)
        self._regressor.fit(polynomial_features, Y)

        # iteratively remove outliers and refit the model on a subset of the data
        errors_px, rmse = self._test_pixel_error(X, Y)
        if outlier_removal_iterations > 0:
            filter_mask = errors_px < self.outlier_threshold_pixel
            X_filtered = X[filter_mask]
            Y_filtered = Y[filter_mask]
            n_filtered_out = X.shape[0] - X_filtered.shape[0]

            if n_filtered_out > 0:
                # if we don't filter anything, we can skip refitting the model here
                logger.debug(
                    f"Fitting. RMSE = {rmse:>7.2f}px ..."
                    f" discarding {n_filtered_out}/{X.shape[0]}"
                    f" ({100 * (n_filtered_out) / X.shape[0]:.2f}%)"
                    f" data points as outliers."
                )
                # recursively remove outliers
                return self.fit(
                    X_filtered,
                    Y_filtered,
                    outlier_removal_iterations=outlier_removal_iterations - 1,
                )

        logger.debug(f"Fitting. RMSE = {rmse:>7.2f}px in final iteration.")
        self._is_fitted = True

    def _test_pixel_error(self, X, Y):
        Y_predict = self.predict(X)
        difference_px = (Y_predict - Y) * self.screen_size
        errors_px = np.linalg.norm(difference_px, axis=1)
        root_mean_squared_error_px = np.sqrt(np.mean(np.square(errors_px)))
        return errors_px, root_mean_squared_error_px

    def predict(self, X):
        self._validate_feature_dimensionality(X)
        polynomial_features = self._polynomial_features(X)
        return self._regressor.predict(polynomial_features)

    def _polynomial_features(self, norm_xy):
        # slice data to retain ndim
        norm_x = norm_xy[:, :1]
        norm_y = norm_xy[:, 1:]

        norm_x_squared = norm_x**2
        norm_y_squared = norm_y**2

        return np.hstack(
            (
                norm_x,
                norm_y,
                norm_x * norm_y,
                norm_x_squared,
                norm_y_squared,
                norm_x_squared * norm_y_squared,
            )
        )

    @staticmethod
    def _validate_reference_dimensionality(Y):
        assert Y.ndim == 2, "Required shape: (n_samples, n_features)"
        assert Y.shape[1] == _REFERENCE_FEATURE_COUNT

    @staticmethod
    @abc.abstractmethod
    def _validate_feature_dimensionality(X):
        raise NotImplementedError


class Model2D_Binocular(Model2D):
    def _polynomial_features(self, norm_xy):
        left = super()._polynomial_features(norm_xy[:, _MONOCULAR_PUPIL_NORM_POS])
        right = super()._polynomial_features(norm_xy[:, _BINOCULAR_PUPIL_NORM_POS])
        return np.hstack((left, right))

    @staticmethod
    def _validate_feature_dimensionality(X):
        assert X.ndim == 2, "Required shape: (n_samples, n_features)"
        assert X.shape[1] == _BINOCULAR_FEATURE_COUNT, (
            f"Received shape: {X.shape}. "
            f"Expected shape (n_samples, {_BINOCULAR_FEATURE_COUNT})"
        )


class Model2D_Monocular(Model2D):
    @staticmethod
    def _validate_feature_dimensionality(X):
        assert X.ndim == 2, "Required shape: (n_samples, n_features)"
        assert X.shape[1] == _MONOCULAR_FEATURE_COUNT, (
            f"Received shape: {X.shape}. "
            f"Expected shape (n_samples, {_MONOCULAR_FEATURE_COUNT})"
        )


class Gazer2D(GazerBase):
    label = "2D"

    @classmethod
    def _gazer_description_text(cls) -> str:
        return "2D gaze mapping: use only in controlled conditions; sensitive to movement of the headset (slippage); uses 2d pupil detection result as input."

    def _init_left_model(self) -> Model:
        return Model2D_Monocular(screen_size=self.g_pool.capture.frame_size)

    def _init_right_model(self) -> Model:
        return Model2D_Monocular(screen_size=self.g_pool.capture.frame_size)

    def _init_binocular_model(self) -> Model:
        return Model2D_Binocular(screen_size=self.g_pool.capture.frame_size)

    def _extract_pupil_features(self, pupil_data) -> np.ndarray:
        pupil_features = np.array([p["norm_pos"] for p in pupil_data])
        assert pupil_features.shape == (len(pupil_data), _MONOCULAR_FEATURE_COUNT)
        return pupil_features

    def _extract_reference_features(self, ref_data) -> np.ndarray:
        ref_features = np.array([r["norm_pos"] for r in ref_data])
        assert ref_features.shape == (len(ref_data), _REFERENCE_FEATURE_COUNT)
        return ref_features

    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)
            gaze_positions = ...  # Placeholder for gaze_positions

            if num_matched == 2:
                if self.binocular_model.is_fitted:
                    right = self._extract_pupil_features([pupil_match[0]])
                    left = self._extract_pupil_features([pupil_match[1]])
                    X = np.hstack([left, right])
                    assert X.shape[1] == _BINOCULAR_FEATURE_COUNT
                    gaze_positions = self.binocular_model.predict(X).tolist()
                    topic = "gaze.2d.01."
                else:
                    logger.debug(
                        "Prediction failed because binocular model is not fitted"
                    )
            elif num_matched == 1:
                X = self._extract_pupil_features([pupil_match[0]])
                assert X.shape[1] == _MONOCULAR_FEATURE_COUNT
                if pupil_match[0]["id"] == 0:
                    if self.right_model.is_fitted:
                        gaze_positions = self.right_model.predict(X).tolist()
                        topic = "gaze.2d.0."
                    else:
                        logger.debug(
                            "Prediction failed because right model is not fitted"
                        )
                elif pupil_match[0]["id"] == 1:
                    if self.left_model.is_fitted:
                        gaze_positions = self.left_model.predict(X).tolist()
                        topic = "gaze.2d.1."
                    else:
                        logger.debug(
                            "Prediction failed because left model is not fitted"
                        )
            else:
                raise ValueError(
                    f"Unexpected number of matched pupil_data: {num_matched}"
                )

            if gaze_positions is ...:
                continue  # Prediction failed and the reason was logged

            for gaze_pos in gaze_positions:
                gaze_datum = {
                    "topic": topic,
                    "norm_pos": gaze_pos,
                    "confidence": np.mean([p["confidence"] for p in pupil_match]),
                    "timestamp": np.mean([p["timestamp"] for p in pupil_match]),
                    "base_data": pupil_match,
                }
                yield gaze_datum

    def filter_pupil_data(
        self, pupil_data: T.Iterable, confidence_threshold: T.Optional[float] = None
    ) -> T.Iterable:
        pupil_data = list(filter(lambda p: "2d" in p["method"], pupil_data))
        pupil_data = super().filter_pupil_data(pupil_data, confidence_threshold)
        return pupil_data
