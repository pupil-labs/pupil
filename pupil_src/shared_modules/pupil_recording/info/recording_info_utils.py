"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import json
import os
import typing as T
import uuid

import csv_utils
from methods import get_system_info

from .recording_info import RecordingInfoFile, Version


def string_from_recording_version(value: Version) -> str:
    return str(value)  # TODO: Make sure this conversion is correct


def string_from_uuid(value: uuid.UUID) -> str:
    return str(value)


def uuid_from_string(value: str) -> uuid.UUID:
    return uuid.UUID(value)


def nanoseconds_from_seconds(value: float) -> int:
    return int(value * 1e9)


def seconds_from_nanoseconds(value: int) -> float:
    return float(value / 1e9)


def default_recording_name(info) -> str:
    if isinstance(info, RecordingInfoFile):
        rec_dir = info.rec_dir
    else:
        rec_dir = str(info)
    return os.path.basename(rec_dir)


def default_system_info(info) -> str:
    return get_system_info()


def validator_version_string(value: str):
    _ = Version(value)


def validator_uuid_string(value: str):
    _ = uuid_from_string(value)


def validator_type(value_type) -> T.Callable[[T.Any], None]:
    def f(value):
        assert isinstance(
            value, value_type
        ), f"Expected instance of type: {value_type} but got: {type(value)}"

    return f


def validator_optional_type(value_type) -> T.Callable[[T.Any], None]:
    def f(value):
        if value is not None:
            validator_type(value_type)(value)

    return f


def read_info_csv_file(rec_dir: str) -> dict:
    file_path = os.path.join(rec_dir, "info.csv")
    with open(file_path, "r") as file:
        return csv_utils.read_key_value_file(file)


def read_info_json_file(rec_dir: str) -> dict:
    file_path = os.path.join(rec_dir, "info.json")
    with open(file_path, "r") as file:
        return json.load(file)


def parse_duration_string(duration_string: str) -> int:
    """Returns number of seconds from string 'HH:MM:SS'."""
    H, M, S = [int(part) for part in duration_string.split(":")]
    SECONDS_PER_H = 3600
    SECONDS_PER_M = 60
    return (H * SECONDS_PER_H) + (M * SECONDS_PER_M) + S
