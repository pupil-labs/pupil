"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as T
import numpy as np

from gaze_mapping.gazer_base import (
    GazerBase,
    Model,
    NotEnoughDataError,
)

from calibration_routines.optimization_calibration.calibrate_2d import (
    calibrate_2d_polynomial,
    make_map_function,
)


class Model2D_v1x(Model):
    def __init__(self, *, screen_size=(1, 1)):
        self.screen_size = screen_size
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def set_params(self, params, map_fn=None):
        self._map_fn = map_fn or make_map_function(*params)
        self._params = params

    def get_params(self):
        return {"params": self._params}


class Model2D_v1x_Binocular(Model2D_v1x):
    def fit(self, X, Y):
        assert X.ndim == Y.ndim == 2, "Required shape: (n_samples, n_features)"
        assert X.shape[0] == Y.shape[0], "Requires same number of samples in X and Y"
        point_cloud = np.hstack([X, Y])
        map_fn, inliers, params = calibrate_2d_polynomial(
            point_cloud, self.screen_size, binocular=True
        )
        if not inliers.any():
            raise NotEnoughDataError

        self.set_params(map_fn=map_fn, params=params)
        self._is_fitted = True

    def predict(self, X):
        # X[:, :2] -> norm_pos left
        # X[:, 2:] -> norm_pos right
        return [self._map_fn(x[:2], x[2:]) for x in X]


class Model2D_v1x_Monocular(Model2D_v1x):
    def fit(self, X, Y):
        assert X.ndim == Y.ndim == 2, "Required shape: (n_samples, n_features)"
        assert X.shape[0] == Y.shape[0], "Requires same number of samples in X and Y"
        point_cloud = np.hstack([X, Y])
        map_fn, inliers, params = calibrate_2d_polynomial(
            point_cloud, self.screen_size, binocular=False
        )
        if not inliers.any():
            raise NotEnoughDataError

        self.set_params(map_fn=map_fn, params=params)
        self._is_fitted = True

    def predict(self, X):
        # X[:, 0:2] -> monocular norm_pos
        return [self._map_fn(x) for x in X]


class Gazer2D_v1x(GazerBase):
    label = "2D (v1)"

    def _init_left_model(self) -> Model:
        return Model2D_v1x_Monocular(screen_size=self.g_pool.capture.frame_size)

    def _init_right_model(self) -> Model:
        return Model2D_v1x_Monocular(screen_size=self.g_pool.capture.frame_size)

    def _init_binocular_model(self) -> Model:
        return Model2D_v1x_Binocular(screen_size=self.g_pool.capture.frame_size)

    def _extract_pupil_features(self, pupil_data) -> np.ndarray:
        pupil_features = np.array([p["norm_pos"] for p in pupil_data])
        assert pupil_features.shape == (len(pupil_data), 2)
        return pupil_features

    def _extract_reference_features(self, ref_data) -> np.ndarray:
        ref_features = np.array([r["norm_pos"] for r in ref_data])
        assert ref_features.shape == (len(ref_data), 2)
        return ref_features

    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)

            if num_matched == 2 and self.binocular_model.is_fitted:
                right = self._extract_pupil_features([pupil_match[0]])
                left = self._extract_pupil_features([pupil_match[1]])
                X = np.hstack([left, right])
                gaze_positions = self.binocular_model.predict(X)
                topic = "gaze.2d.01."
            elif num_matched == 1:
                X = self._extract_pupil_features([pupil_match[0]])
                if pupil_match[0]["id"] == 0 and self.right_model.is_fitted:
                    gaze_positions = self.right_model.predict(X)
                    topic = "gaze.2d.0."
                elif pupil_match[0]["id"] == 1 and self.left_model.is_fitted:
                    gaze_positions = self.left_model.predict(X)
                    topic = "gaze.2d.1."

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
