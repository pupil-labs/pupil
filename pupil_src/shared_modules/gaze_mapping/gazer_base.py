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
import typing as T

from plugin import Plugin
from calibration_routines import data_processing

from .matching import RealtimeMatcher


class NotEnoughDataError(Exception):
    pass


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y):
        """Fit model with input `X` to targets `y`

        Arguments:
            X {array-like} -- of shape (n_samples, n_features)
            y {array-like} -- of shape (n_samples,) or (n_samples, n_targets)
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
    def _fit_monocular_model(self, model: Model, matched_data: T.Iterable):
        """Calls `model.fit()` with appropriate data from `matched_data`

        Arguments:
            model {Model} -- Model to fit
            matched_data {T.Iterable} -- List of matched data
        """
        pass

    @abc.abstractmethod
    def _fit_binocular_model(self, model: Model, matched_data: T.Iterable):
        """Calls `model.fit()` with appropriate data from `matched_data`

        Arguments:
            model {Model} -- Model to fit
            matched_data {T.Iterable} -- List of matched data
        """
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
        if confidence_threshold is not None:
            pupil_data = data_processing._filter_pupil_list_by_confidence(
                pupil_data, confidence_threshold
            )
        return pupil_data

    # ------------ Base Implementation

    # -- Plugin Functions

    def __init__(self, g_pool, *, calib_data=None, params=None):
        super().__init__(g_pool)
        if None not in (calib_data, params):
            raise ValueError("`calib_data` and `params` are mutually exclusive")

        self.init_models()
        self.init_matcher()

        if calib_data is not None:
            self.fit_on_calib_data(calib_data)
        elif params is not None:
            self.set_params(params)
        else:
            raise ValueError("Requires either `calib_data` or `params`")

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
            "left_model": self.left_model.get_params(),
            "right_model": self.right_model.get_params(),
            "binocular_model": self.binocular_model.get_params(),
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

        self._fit_monocular_model(self.left_model, matches.left)
        self._fit_monocular_model(self.right_model, matches.right)
        self._fit_binocular_model(self.binocular_model, matches.binocular)

    def match_pupil_to_ref(self, pupil_data, ref_data) -> "Matches":
        matches = data_processing._match_data(pupil_data, ref_data)
        bino, _, right, left = matches
        matches = Matches(left, right, bino)
        return matches

    def map_pupil_to_gaze(self, pupil_data, sort_by_creation_time=True):
        pupil_data = self.filter_pupil_data(pupil_data)
        if sort_by_creation_time:
            pupil_data.sort(key=lambda p: p["timestamp"])

        matches = (self.matcher.on_pupil_datum(datum) for datum in pupil_data)
        yield from self.predict(matches)


class Matches(T.NamedTuple):
    left: object
    right: object
    binocular: object
