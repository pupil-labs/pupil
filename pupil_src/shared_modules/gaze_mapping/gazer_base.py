"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import itertools
import logging
import typing as T

import numpy as np

from plugin import Plugin
from calibration_routines import data_processing

from .matching import RealtimeMatcher

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    pass


class NotEnoughDataError(CalibrationError):
    message = "Not sufficient data available."


class FitDidNotConvergeError(CalibrationError):
    message = "Model fit did not converge."


class Model(abc.ABC):
    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit model with input `X` to targets `y`

        Arguments:
            X {array-like} -- of shape (n_samples, n_features)
            y {array-like} -- of shape (n_samples, n_targets)
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Predict values based on input `X`

        Arguments:
            X {array-like} -- of shape (n_samples, n_features)

        Returns:
            array-like -- of shape (n_samples, n_outputs)
        """
        pass

    @abc.abstractmethod
    def set_params(self, **params):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass


class GazerBase(abc.ABC, Plugin):
    label: str = ...  # Subclasses should set this to a meaningful name
    uniqueness = "by_base_class"

    @abc.abstractmethod
    def _init_left_model(self) -> Model:
        pass

    @abc.abstractmethod
    def _init_right_model(self) -> Model:
        pass

    @abc.abstractmethod
    def _init_binocular_model(self) -> Model:
        pass

    @abc.abstractmethod
    def _extract_pupil_features(self, pupil_data) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _extract_reference_features(self, ref_data) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict(
        self, matched_pupil_data: T.Iterator[T.List["Pupil"]]
    ) -> T.Iterator["Gaze"]:
        pass

    def filter_pupil_data(
        self, pupil_data: T.Iterable, confidence_threshold: T.Optional[float] = None
    ) -> T.Iterable:
        """Filters pupil data by confidence

        Overwrite to extend filter functionality

        Arguments:
            pupil_data {T.Iterable} -- [description]

        Keyword Arguments:
            confidence_threshold {T.Optional[float]} -- (default: {None})

        Returns:
            T.Iterable -- Subset of `pupil_data`
        """
        # TODO: make this a generator
        if confidence_threshold is not None:
            pupil_data = data_processing._filter_pupil_list_by_confidence(
                pupil_data, confidence_threshold
            )
        return pupil_data

    # ------------ Base Implementation

    # -- Plugin Functions

    def __init__(
        self, g_pool, *, calib_data=None, params=None, raise_calibration_error=False
    ):
        super().__init__(g_pool)
        if None not in (calib_data, params):
            raise ValueError("`calib_data` and `params` are mutually exclusive")

        self.init_models()
        self.init_matcher()

        if calib_data is not None:
            try:
                self.fit_on_calib_data(calib_data)
            except CalibrationError:
                if raise_calibration_error:
                    raise  # Let offline calibration handle this one!
                logger.error("Calibration Failed!")
                self.alive = False
                note = self.create_calibration_failed_notification(err)
                self.notify_all(note)
            except Exception as err:
                raise CalibrationError from err
            else:
                self.notify_all(self.create_successful_notification())
        elif params is not None:
            self.set_params(params)
        else:
            raise ValueError("Requires either `calib_data` or `params`")

    def create_successful_notification(self):
        return {
            "subject": "calibration.successful",
            "method": self.__class__.__name__,
            "timestamp": self.g_pool.get_timestamp(),
            "record": True,
        }

    def create_calibration_failed_notification(self, error):
        msg = error.__class__.__name__
        logger.error(msg)
        return {
            "subject": "calibration.failed",
            "reason": msg,
            "timestamp": self.g_pool.get_timestamp(),
            "record": True,
        }

    def init_ui(self):
        self.add_menu()

    def deinit_ui(self):
        self.remove_menu()

    def get_init_dict(self):
        return {"params": self.get_params()}

    def recent_events(self, events):
        pupil_data = events["pupil"]
        recent_gaze_data = []
        for gaze in self.map_pupil_to_gaze(pupil_data):
            # TODO: publish on network
            recent_gaze_data.append(gaze)
        events["gaze"] = recent_gaze_data

    # -- Core Functionality

    def get_params(self):
        return {
            "left_model": dict(self.left_model.get_params()),
            "right_model": dict(self.right_model.get_params()),
            "binocular_model": dict(self.binocular_model.get_params()),
        }

    def set_params(self, params):
        self.left_model.set_params(**params["left_model"])
        self.right_model.set_params(**params["right_model"])
        self.binocular_model.set_params(**params["binocular_model"])

    def init_models(self):
        self.left_model: Model = self._init_left_model()
        self.right_model: Model = self._init_right_model()
        self.binocular_model: Model = self._init_binocular_model()

    def init_matcher(self):
        self.matcher = RealtimeMatcher()

    def fit_on_calib_data(self, calib_data):
        # extract reference data
        ref_data = calib_data["ref_list"]
        # extract and filter pupil data
        pupil_data = calib_data["pupil_list"]
        pupil_data = self.filter_pupil_data(
            pupil_data, self.g_pool.min_calibration_confidence
        )
        # match pupil to reference data (left, right, and binocular)
        matches = self.match_pupil_to_ref(pupil_data, ref_data)
        if matches.binocular[0]:
            self._fit_binocular_model(self.binocular_model, matches.binocular)
            self._fit_monocular_model(self.right_model, matches.right)
            self._fit_monocular_model(self.left_model, matches.left)
        elif matches.right[0]:
            self._fit_monocular_model(self.right_model, matches.right)
        elif matches.left[0]:
            self._fit_monocular_model(self.left_model, matches.left)
        else:
            raise NotEnoughDataError

    def _fit_binocular_model(self, model: Model, matched_data: T.Iterable):
        X, Y = self.extract_features_from_matches_binocular(matched_data)
        model.fit(X, Y)

    def _fit_monocular_model(self, model: Model, matched_data: T.Iterable):
        X, Y = self.extract_features_from_matches_monocular(matched_data)
        model.fit(X, Y)

    def match_pupil_to_ref(self, pupil_data, ref_data) -> "Matches":
        matches = data_processing._match_data_batch(pupil_data, ref_data)
        bino, right, left = matches
        matches = Matches(left, right, bino)
        return matches

    def extract_features_from_matches_binocular(self, binocular_matches):
        ref, pupil_right, pupil_left = binocular_matches
        Y = self._extract_reference_features(ref)
        X_right = self._extract_pupil_features(pupil_right)
        X_left = self._extract_pupil_features(pupil_left)
        X = np.hstack([X_left, X_right])
        return X, Y

    def extract_features_from_matches_monocular(self, monocular_matches):
        ref, pupil = monocular_matches
        Y = self._extract_reference_features(ref)
        X = self._extract_pupil_features(pupil)
        return X, Y

    def map_pupil_to_gaze(self, pupil_data, sort_by_creation_time=True):
        pupil_data = self.filter_pupil_data(pupil_data)
        if sort_by_creation_time:
            pupil_data.sort(key=lambda p: p["timestamp"])

        matches = (self.matcher.on_pupil_datum(datum) for datum in pupil_data)
        matches = itertools.chain.from_iterable(matches)

        yield from self.predict(matches)


class Matches(T.NamedTuple):
    left: object
    right: object
    binocular: object
