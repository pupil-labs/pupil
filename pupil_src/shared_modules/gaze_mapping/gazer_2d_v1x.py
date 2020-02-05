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
    data_processing,
    NotEnoughDataError,
)

from calibration_routines.optimization_calibration.calibrate_2d import (
    calibrate_2d_polynomial,
    make_map_function,
)


class Model2D_v1x(Model):
    def __init__(self, *, binocular, screen_size=(1, 1)):
        self.binocular = binocular
        self.screen_size = screen_size
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, point_cloud):  # NOTE: breaks api due to legacy reasons
        map_fn, inliers, params = calibrate_2d_polynomial(
            point_cloud, self.screen_size, binocular=self.binocular
        )
        if not inliers.any():
            raise NotEnoughDataError

        self.set_params(map_fn=map_fn, params=params)
        self._is_fitted = True

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        if self.binocular:
            p0, p1 = x
            return self._map_fn(p0["norm_pos"], p1["norm_pos"])
        else:
            return self._map_fn(x[0]["norm_pos"])

    def set_params(self, params, map_fn=None):
        self._map_fn = map_fn or make_map_function(*params)
        self._params = params

    def get_params(self):
        return {"params": self._params}


class Gazer2D_v1x(GazerBase):
    label = "2D (v1)"

    def _init_left_model(self) -> Model:
        return Model2D_v1x(binocular=False, screen_size=self.g_pool.capture.frame_size)

    def _init_right_model(self) -> Model:
        return Model2D_v1x(binocular=False, screen_size=self.g_pool.capture.frame_size)

    def _init_binocular_model(self) -> Model:
        return Model2D_v1x(binocular=True, screen_size=self.g_pool.capture.frame_size)

    def _fit_monocular_model(self, model: Model, matched_data: T.Iterable):
        point_cloud = data_processing._extract_2d_data_monocular(matched_data)
        if not point_cloud:
            raise NotEnoughDataError
        model.fit(point_cloud)

    def _fit_binocular_model(self, model: Model, matched_data: T.Iterable):
        point_cloud = data_processing._extract_2d_data_binocular(matched_data)
        if not point_cloud:
            raise NotEnoughDataError
        model.fit(point_cloud)

    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)

            if num_matched == 2 and self.binocular_model.is_fitted:
                gaze_positions = self.binocular_model.predict([pupil_match])
                topic = "gaze.2d.01."
            elif num_matched == 1:
                if pupil_match[0]["id"] == 0 and self.right_model.is_fitted:
                    gaze_positions = self.right_model.predict([pupil_match])
                    topic = "gaze.2d.0."
                elif pupil_match[0]["id"] == 1 and self.left_model.is_fitted:
                    gaze_positions = self.left_model.predict([pupil_match])
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
        pupil_data = list(filter(lambda p: "2d" in p["topic"], pupil_data))
        pupil_data = super().filter_pupil_data(pupil_data, confidence_threshold)
        return pupil_data
