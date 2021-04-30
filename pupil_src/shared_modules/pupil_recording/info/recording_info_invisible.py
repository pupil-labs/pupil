import pathlib
import uuid

from version_utils import ParsedVersion, parse_version

from . import recording_info_utils as utils
from .recording_info import RecordingInfoFile


class InvisibleRecordingInfoFile(RecordingInfoFile):

    # Public

    file_name = "info.json"

    # RecordingInfo
    @property
    def file_path(self) -> str:
        return str(pathlib.Path(self.rec_dir) / self.file_name)

    @property
    def meta_version(self) -> ParsedVersion:
        return parse_version("1.0")

    @property
    def min_player_version(self) -> ParsedVersion:
        return parse_version("3.2")

    @property
    def recording_uuid(self) -> uuid.UUID:
        return utils.uuid_from_string(self["recording_id"])

    @recording_uuid.setter
    def recording_uuid(self, value: uuid.UUID):
        self["recording_id"] = utils.string_from_uuid(value)

    # start_time_system: Unix epoch
    @property
    def start_time_system_s(self) -> float:
        return utils.seconds_from_nanoseconds(self.start_time_system_ns)

    @start_time_system_s.setter
    def start_time_system_s(self, value: float):
        self.start_time_system_ns = utils.nanoseconds_from_seconds(value)

    @property
    def start_time_system_ns(self) -> int:
        return self["start_time"]

    @start_time_system_ns.setter
    def start_time_system_ns(self, value: int):
        self["start_time"] = value

    # start_time_system infers start_time_synced
    @property
    def start_time_synced_s(self) -> float:
        return self.start_time_system_s

    @start_time_synced_s.setter
    def start_time_synced_s(self, value: float):
        self.start_time_system_s = value

    @property
    def start_time_synced_ns(self) -> int:
        return self.start_time_system_ns

    @start_time_synced_ns.setter
    def start_time_synced_ns(self, value: int):
        self.start_time_system_ns = value

    @property
    def duration_s(self) -> float:
        return utils.seconds_from_nanoseconds(self.duration_ns)

    @duration_s.setter
    def duration_s(self, value: float):
        self.duration_ns = utils.nanoseconds_from_seconds(value)

    @property
    def duration_ns(self) -> int:
        return self["duration"]

    @duration_ns.setter
    def duration_ns(self, value: int):
        self["duration"] = value

    @property
    def recording_software_name(self) -> str:
        return self.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE

    @recording_software_name.setter
    def recording_software_name(self, value: str):
        self["recording_software_name"] = str(value)

    @property
    def recording_software_version(self) -> str:
        return self["app_version"]

    @recording_software_version.setter
    def recording_software_version(self, value: str):
        self["app_version"] = value

    @property
    def recording_name(self) -> str:
        try:
            return str(self["template_data"]["recording_name"])
        except KeyError:
            return utils.default_recording_name(self)

    @recording_name.setter
    def recording_name(self, value: str):
        self["recording_name"] = str(value)

    @property
    def system_info(self) -> str:
        try:
            return str(self["android_device_name"])
        except KeyError:
            return utils.default_system_info(self)

    @system_info.setter
    def system_info(self, value: str):
        self["android_device_name"] = str(value)

    # RecordingInfoFile

    @property
    def _private_key_schema(self) -> RecordingInfoFile._KeyValueSchema:
        return {}
