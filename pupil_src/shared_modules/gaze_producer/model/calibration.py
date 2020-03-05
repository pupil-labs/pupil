"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import namedtuple
import copy
import typing as T

import file_methods as fm

from storage import StorageItem

# this plugin does not care about the content of the result, it just receives it from
# the calibration routine and handles it to the gaze mapper
CalibrationSetup = namedtuple("CalibrationResult", ["gazer_class_name", "calib_data"])
CalibrationResult = namedtuple("CalibrationResult", ["gazer_class_name", "params"])


class Calibration(StorageItem):
    version = 2

    def __init__(
        self,
        *,
        unique_id,
        name,
        recording_uuid,
        gazer_class_name,
        frame_index_range,
        minimum_confidence,
        status: str,
        is_offline_calibration: bool,
        calib_data: T.Optional[T.Any] = None,
        calib_params: T.Optional[T.Any] = None,
    ):
        # Set arbitrarily mutable properties
        self.name = name
        self.unique_id = unique_id
        self.recording_uuid = recording_uuid
        self.gazer_class_name = gazer_class_name
        self.frame_index_range = frame_index_range
        self.minimum_confidence = minimum_confidence
        self.status = status

        # Set immutable properties or properties that must be mutated in a consistent way
        self.__is_offline_calibration = is_offline_calibration
        self.__calib_data = calib_data
        self.__calib_params = calib_params

        # Assert all properties are consistent
        try:
            self.__assert_property_consistency()
        except ValueError:
            raise
        except Exception as err:
            raise ValueError(str(err))

    @property
    def is_offline_calibration(self) -> bool:
        return self.__is_offline_calibration

    @property
    def calib_data(self) -> T.Optional[T.Any]:
        return self.__calib_data

    @property
    def calib_params(self) -> T.Optional[T.Any]:
        return self.__calib_params

    @property
    def params(self) -> T.Optional[T.Any]:
        """Alias for calib_params"""
        return self.calib_params

    def update(
        self,
        is_offline_calibration: bool = ...,
        calib_data: T.Optional[T.Any] = ...,
        calib_params: T.Optional[T.Any] = ...,
    ):
        if is_offline_calibration is not ...:
            self.__is_offline_calibration = is_offline_calibration
        if calib_data is not ...:
            self.__calib_data = calib_data
        if calib_params is not ...:
            self.__calib_params = calib_params
        try:
            self.__assert_property_consistency()
        except ValueError:
            raise
        except Exception as err:
            raise ValueError(str(err))

    @staticmethod
    def from_dict(dict_: dict) -> "Calibration":
        try:
            if dict_["version"] != Calibration.version:
                raise ValueError(f"Model version missmatch")
            del dict_["version"]
            return Calibration(**dict_)
        except (KeyError, ValueError, TypeError) as err:
            raise ValueError(str(err))

    @property
    def as_dict(self) -> dict:
        self.__assert_property_consistency()  # sanity check
        dict_ = {k: v(self) for (k, v) in self.__schema}
        dict_ = fm._recursive_deep_copy(dict_)
        return dict_

    @staticmethod
    def from_tuple(tuple_):
        keys = [k for (k, _) in Calibration.__schema]
        dict_ = dict(zip(keys, tuple_))
        return Calibration.from_dict(dict_)

    @property
    def as_tuple(self):
        keys = [k for (k, _) in Calibration.__schema]
        dict_ = self.as_dict
        return tuple(dict_[k] for k in keys)

    ### Private

    __schema = (
        ("version", lambda self: self.version),
        ("unique_id", lambda self: self.unique_id),
        ("name", lambda self: self.name),
        ("recording_uuid", lambda self: self.recording_uuid),
        ("gazer_class_name", lambda self: self.gazer_class_name),
        ("frame_index_range", lambda self: self.frame_index_range),
        ("minimum_confidence", lambda self: self.minimum_confidence),
        ("status", lambda self: self.status),
        ("is_offline_calibration", lambda self: self.__is_offline_calibration),
        ("calib_data", lambda self: self.__calib_data),
        ("calib_params", lambda self: self.__calib_params),
    )

    def __assert_property_consistency(self):
        if self.__is_offline_calibration:
            if self.__calib_data is not None:
                raise ValueError(
                    f"Unexpected calib_data argument for offline calibration"
                )
        else:
            if self.__calib_params is not None:
                raise ValueError(
                    f"Unexpected calib_params argument for pre-recorded calibration"
                )
            if self.__calib_data is None:
                raise ValueError(
                    f"Expected calib_data argument for pre-recorded calibration"
                )
