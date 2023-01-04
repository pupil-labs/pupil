"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import uuid

from version_utils import ParsedVersion, parse_version

from . import RecordingInfoFile
from . import recording_info_utils as utils


class _RecordingInfoFile_2_0(RecordingInfoFile):

    # RecordingInfo

    @property
    def meta_version(self) -> ParsedVersion:
        return parse_version("2.0")

    @property
    def min_player_version(self) -> ParsedVersion:
        return parse_version("1.16")

    @property
    def recording_uuid(self) -> uuid.UUID:
        return utils.uuid_from_string(self["recording_uuid"])

    @recording_uuid.setter
    def recording_uuid(self, value: uuid.UUID):
        self["recording_uuid"] = utils.string_from_uuid(value)

    @property
    def start_time_system_s(self) -> float:
        return float(self["start_time_system_s"])

    @start_time_system_s.setter
    def start_time_system_s(self, value: float):
        self["start_time_system_s"] = float(value)

    @property
    def start_time_system_ns(self) -> int:
        return utils.nanoseconds_from_seconds(self.start_time_system_s)

    @start_time_system_ns.setter
    def start_time_system_ns(self, value: int):
        self.start_time_system_s = utils.seconds_from_nanoseconds(value)

    @property
    def start_time_synced_s(self) -> float:
        return float(self["start_time_synced_s"])

    @start_time_synced_s.setter
    def start_time_synced_s(self, value: float):
        self["start_time_synced_s"] = float(value)

    @property
    def start_time_synced_ns(self) -> int:
        return utils.nanoseconds_from_seconds(self.start_time_synced_s)

    @start_time_synced_ns.setter
    def start_time_synced_ns(self, value: int):
        self.start_time_synced_s = utils.seconds_from_nanoseconds(value)

    @property
    def duration_s(self) -> float:
        return float(self["duration_s"])

    @duration_s.setter
    def duration_s(self, value: float):
        self["duration_s"] = float(value)

    @property
    def duration_ns(self) -> int:
        return utils.nanoseconds_from_seconds(self.duration_s)

    @duration_ns.setter
    def duration_ns(self, value: int):
        self.duration_s = utils.seconds_from_nanoseconds(value)

    @property
    def recording_software_name(self) -> str:
        return str(self["recording_software_name"])

    @recording_software_name.setter
    def recording_software_name(self, value: str):
        self["recording_software_name"] = str(value)

    @property
    def recording_software_version(self) -> str:
        return self["recording_software_version"]

    @recording_software_version.setter
    def recording_software_version(self, value: str):
        self["recording_software_version"] = value

    @property
    def recording_name(self) -> str:
        try:
            return str(self["recording_name"])
        except KeyError:
            return utils.default_recording_name(self)

    @recording_name.setter
    def recording_name(self, value: str):
        self["recording_name"] = str(value)

    @property
    def system_info(self) -> str:
        try:
            return str(self["system_info"])
        except KeyError:
            return utils.default_system_info(self)

    @system_info.setter
    def system_info(self, value: str):
        self["system_info"] = str(value)

    # RecordingInfoFile

    @property
    def _private_key_schema(self) -> RecordingInfoFile._KeyValueSchema:
        return {
            "meta_version": (utils.validator_version_string, lambda _: "2.0"),
            "min_player_version": (utils.validator_version_string, lambda _: "1.16"),
            "recording_uuid": (utils.validator_uuid_string, lambda _: uuid.uuid4()),
            "start_time_system_s": (utils.validator_type(float), None),
            "start_time_synced_s": (utils.validator_type(float), None),
            "duration_s": (utils.validator_type(float), None),
            "recording_software_name": (utils.validator_type(str), None),
            "recording_software_version": (utils.validator_type(str), None),
            "recording_name": (utils.validator_type(str), utils.default_recording_name),
            "system_info": (utils.validator_type(str), utils.default_system_info),
        }
