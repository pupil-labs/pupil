"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from __future__ import annotations

import sys
import typing as T

from typing_extensions import Self, TypedDict

if sys.version_info < (3, 9):

    def get_type_hints(cls):
        return cls.__annotations__

else:
    # Starting in Python 3.10, __annotations__ does no longer contain the annotations
    # of the parent class. `_SerializedNamedTupleMixin.sanitize_serialized_dict`
    # depends on the earlier implementation.
    # We do not use `inspect.get_annotations()` as a replacement for two reasons:
    # 1. It also does not include the base classes' annoations
    # 2. The annotations might not be evaluated yet, containing `typing.ForwardRef`
    #    instances
    #
    # `typing.get_type_hints` implements the desired behavior but is only available in
    # Python 3.9 or newer
    from typing import get_type_hints


# TODO: Consider extending this pattern for notifications through the entire codebase and/or replace with dataclasses with Python 3.7


class _SerializedNamedTupleMixin:
    def as_dict(self) -> dict:
        return dict(self._asdict())

    @classmethod
    def from_dict(cls, dict_: dict) -> Self:
        dict_ = {**cls._field_defaults, **dict_}
        try:
            dict_ = cls.sanitize_serialized_dict(dict_)
            return cls(**dict_)
        except Exception as err:
            raise ValueError(f"{err}") from err

    @classmethod
    def sanitize_serialized_dict(cls, dict_: dict) -> dict:
        field_classes = get_type_hints(cls)
        for field_name in cls._fields:
            field_cls = field_classes[field_name]
            dict_[field_name] = field_cls(dict_[field_name])
        return dict_

    @classmethod
    def _assert_static_property_matches_dict(
        cls,
        dict_: dict,
        key_name: str,
        key_type: T.Any,
        field_name: str = None,
    ):
        field_name = field_name if field_name is not None else key_name
        defaults = cls._field_defaults

        # Assert values present in dict_ and cls
        assert key_name in dict_, f'Serialized dict must contain "{key_name}" key'
        assert field_name in defaults, f"{cls.__name__} must define {field_name} value"

        # Extract the dict_ and cls values
        dict_val = dict_[key_name]
        real_val = defaults[field_name]

        # Assert dict_ and cls value types
        assert isinstance(
            dict_val, key_type
        ), f'Serialized dict value for "{key_name}" must be of type {key_type.__name__}'
        assert isinstance(
            real_val, key_type
        ), f"{cls.__name__} value for {field_name} must be of type {key_type.__name__}"

        # Assert dict_ and cls values are equal
        assert (
            dict_val == real_val
        ), f"{key_name} missmatch: expected {real_val}, but got {dict_val}"


class _NotificationMixin(_SerializedNamedTupleMixin):
    @classmethod
    def sanitize_serialized_dict(cls, dict_: dict) -> dict:
        cls._assert_static_property_matches_dict(dict_, "subject", str)
        if "topic" in dict_:
            del dict_["topic"]
        return super().sanitize_serialized_dict(dict_)


class _VersionedNotificationMixin(_NotificationMixin):
    @classmethod
    def sanitize_serialized_dict(cls, dict_: dict) -> dict:
        cls._assert_static_property_matches_dict(dict_, "version", int)
        return super().sanitize_serialized_dict(dict_)


### Notification fields


class _CalibrationSuccessFields(T.NamedTuple):
    gazer_class_name: str
    timestamp: float
    record: bool = False

    # Meta
    subject: str = f"calibration.successful"


class _CalibrationFailureFields(T.NamedTuple):
    reason: str
    gazer_class_name: str
    timestamp: float
    record: bool = False

    # Meta
    subject: str = f"calibration.failed"


class CalibrationData(TypedDict):
    ref_list: T.List
    pupil_list: T.List


class _CalibrationSetupFields(T.NamedTuple):
    gazer_class_name: str
    timestamp: float
    calib_data: CalibrationData
    record: bool = False

    # Meta
    version: int = 2
    subject: str = f"calibration.setup.v{version}"


class _CalibrationResultFields(T.NamedTuple):
    gazer_class_name: str
    timestamp: float
    params: dict
    record: bool = False

    # Meta
    version: int = 2
    subject: str = f"calibration.result.v{version}"


### Notifications


class CalibrationSuccessNotification(_CalibrationSuccessFields, _NotificationMixin):
    pass


class CalibrationFailureNotification(_CalibrationFailureFields, _NotificationMixin):
    pass


class CalibrationSetupNotification(
    _CalibrationSetupFields, _VersionedNotificationMixin
):
    @classmethod
    def calibration_format_version(cls) -> int:
        return cls._field_defaults["version"]

    @staticmethod
    def file_name() -> str:
        return f"prerecorded_calibration_setup"


class CalibrationResultNotification(
    _CalibrationResultFields, _VersionedNotificationMixin
):
    @classmethod
    def calibration_format_version(cls) -> int:
        return cls._field_defaults["version"]

    @staticmethod
    def file_name() -> str:
        return f"prerecorded_calibration_result"
