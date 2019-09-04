import abc
import collections
import csv
import datetime
import json
import math
import os
import re
import time
import typing as T
import uuid
from packaging.version import Version as RecordingVersion

import csv_utils


class RecordingInfoInvalidError(Exception):
    @staticmethod
    def missingKey(key: str) -> "RecordingInfoInvalidError":
        return RecordingInfoInvalidError(f"Key \"{key}\" is missing")

    @staticmethod
    def wrongTypeForKey(key: str, actual_type, expected_type) -> "RecordingInfoInvalidError":
        return RecordingInfoInvalidError(
            f"Value for key \"{key}\" is of the wrong type \"{actual_type}\"; expected \"{expected_type}\"")


class RecordingInfo(collections.abc.MutableMapping):

    @property
    @abc.abstractmethod
    def recording_uuid(self) -> uuid.UUID:
        pass

    @recording_uuid.setter
    @abc.abstractmethod
    def recording_uuid(self, value: uuid.UUID):
        pass

    @property
    @abc.abstractmethod
    def recording_name(self) -> str:
        pass

    @recording_name.setter
    @abc.abstractmethod
    def recording_name(self, value: str):
        pass

    @property
    @abc.abstractmethod
    def software_version(self) -> RecordingVersion:
        pass

    @software_version.setter
    @abc.abstractmethod
    def software_version(self, value: RecordingVersion):
        pass

    @property
    @abc.abstractmethod
    def data_format_version(self) -> T.Optional[RecordingVersion]:
        pass

    @data_format_version.setter
    @abc.abstractmethod
    def data_format_version(self, value: T.Optional[RecordingVersion]):
        pass

    @property
    @abc.abstractmethod
    def duration_s(self) -> float:
        pass

    @duration_s.setter
    @abc.abstractmethod
    def duration_s(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def duration_ns(self) -> int:
        pass

    @duration_ns.setter
    @abc.abstractmethod
    def duration_ns(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def start_time_s(self) -> float:
        pass

    @start_time_s.setter
    @abc.abstractmethod
    def start_time_s(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def start_time_ns(self) -> int:
        pass

    @start_time_ns.setter
    @abc.abstractmethod
    def start_time_ns(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def start_time_synced_s(self) -> float:
        pass

    @start_time_synced_s.setter
    @abc.abstractmethod
    def start_time_synced_s(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def start_time_synced_ns(self) -> int:
        pass

    @start_time_synced_ns.setter
    @abc.abstractmethod
    def start_time_synced_ns(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def world_camera_frames(self) -> int:
        pass

    @world_camera_frames.setter
    @abc.abstractmethod
    def world_camera_frames(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def world_camera_resolution(self) -> T.Tuple[int, int]:
        pass

    @world_camera_resolution.setter
    @abc.abstractmethod
    def world_camera_resolution(self, value: T.Tuple[int, int]):
        pass

    ValueValidation = T.Callable[[T.Any], T.Optional[str]]
    ValueDefaultGetter = T.Callable[["RecordingInfo"], T.Any]

    @property
    @abc.abstractmethod
    def _schema(self) -> T.Mapping[str, T.Tuple[ValueValidation, T.Optional[ValueDefaultGetter]]]:
        pass

    def validate(self):
        for key, (validation, default_getter) in self._schema.items():
            if key not in self:
                if default_getter is None:
                    raise RecordingInfoInvalidError.missingKey(key)
                else:
                    self[key] = default_getter(self)
                    continue
            validation_error = validation(self[key])
            if validation_error is not None:
                raise RecordingInfoInvalidError(f"Error on key \"{key}\": {validation_error}")

    @staticmethod
    def property_equality(x: "RecordingInfo", y: "RecordingInfo") -> bool:

        def _equal_sec(x: float, y: float) -> bool:
            # Decimal precision is lost when writing to / reading from info.csv
            return math.floor(x) == math.floor(y)

        return  x.recording_uuid == y.recording_uuid \
            and x.recording_name == y.recording_name \
            and x.software_version == y.software_version \
            and x.data_format_version == y.data_format_version \
            and x.world_camera_frames == y.world_camera_frames \
            and x.world_camera_resolution == y.world_camera_resolution \
            and _equal_sec(x.duration_s, y.duration_s) \
            and _equal_sec(x.start_time_s, y.start_time_s) \
            and _equal_sec(x.start_time_synced_s, y.start_time_synced_s)

    def copy_from(self, other):
        self.recording_uuid = other.recording_uuid
        self.recording_name = other.recording_name
        self.software_version = other.software_version
        self.data_format_version = other.data_format_version
        self.duration_ns = other.duration_ns
        self.start_time_ns = other.start_time_ns
        self.start_time_synced_ns = other.start_time_synced_ns
        self.world_camera_frames = other.world_camera_frames
        self.world_camera_resolution = other.world_camera_resolution
        assert RecordingInfo.property_equality(self, other)


class RecordingInfoFile(RecordingInfo):

    @property
    @abc.abstractmethod
    def file_name(self) -> str:
        pass

    @staticmethod
    @abc.abstractmethod
    def _read_dict_from_file(file) -> dict:
        pass

    @staticmethod
    @abc.abstractmethod
    def _write_dict_to_file(file, dict_to_write: dict, sort_keys: bool):
        pass

    def __init__(self, rec_dir: str, should_load_file: bool = True, should_validate: bool = True):
        self._rec_dir = str(rec_dir)
        if should_load_file:
            self.load_file(should_validate=should_validate)

    @property
    def rec_dir(self) -> str:
        return self._rec_dir

    @property
    def file_path(self) -> str:
        return os.path.join(self.rec_dir, self.file_name)

    def _keys_to_save(self):
        return set(self._schema.keys())

    def _dict_to_save(self):
        dict_to_save = {}
        for key in self._keys_to_save():
            try:
                dict_to_save[key] = self[key]
            except KeyError:
                raise RecordingInfoInvalidError.missingKey(key=key)
        return dict_to_save

    def save_file(self, should_validate: bool = True, sort_keys: bool = True):
        if should_validate:
            self.validate()
        with open(self.file_path, "w") as file:
            self._write_dict_to_file(
                file=file,
                dict_to_write=self._dict_to_save(),
                sort_keys=sort_keys
            )

    def load_file(self, should_validate: bool = True):
        with open(self.file_path, "r") as file:
            read_dict = self._read_dict_from_file(file=file)
        self.update(read_dict)
        if should_validate:
            self.validate()

    def _get_cached_computed_property(self, key: str, getter, *getter_args, **getter_kwargs):
        try:
            return self[key]
        except KeyError:
            cached_value = getter(*getter_args, **getter_kwargs)
            self._set_cached_computed_property(key=key, cached_value=cached_value)
            return cached_value

    def _set_cached_computed_property(self, key: str, cached_value):
        self[key] = cached_value

    # MutableMapping

    def __getitem__(self, key):
        return self.__info.__getitem__(key)

    def __setitem__(self, key, item):
        return self.__info.__setitem__(key, item)

    def __delitem__(self, key):
        return self.__info.__delitem__(key)

    def __iter__(self):
        return self.__info.__iter__()

    def __len__(self):
        return self.__info.__len__()

    @property
    def __info(self) -> dict:
        try:
            return self.__info_storage
        except AttributeError:
            self.__info_storage = {}
            return self.__info_storage


class RecordingInfoFileCSV(RecordingInfoFile):

    # RecordingInfo

    @property
    def recording_uuid(self) -> uuid.UUID:
        return uuid.UUID(self["Recording UUID"])

    @recording_uuid.setter
    def recording_uuid(self, value: uuid.UUID):
        self["Recording UUID"] = str(value)

    @property
    def recording_name(self) -> str:
        return self["Recording Name"]

    @recording_name.setter
    def recording_name(self, value: str):
        self["Recording Name"] = str(value)

    @property
    def software_version(self) -> RecordingVersion:
        return RecordingVersion(self["Capture Software Version"])

    @software_version.setter
    def software_version(self, value: RecordingVersion):
        self["Capture Software Version"] = str(value) #TODO: Test if this conversion is correct

    @property
    def data_format_version(self) -> T.Optional[RecordingVersion]:
        version_string = self["Data Format Version"]
        if version_string:
            return RecordingVersion(version_string)
        else:
            return None

    @data_format_version.setter
    def data_format_version(self, value: T.Optional[RecordingVersion]):
        self["Data Format Version"] = str(value) if value else None #TODO: Test if this conversion is correct

    @property
    def duration_s(self) -> float:
        return _parse_time_string_to_seconds(time_str=self["Duration Time"])

    @duration_s.setter
    def duration_s(self, value: float):
        self["Duration Time"] = _format_seconds_to_time_string(sec=value)

    @property
    def duration_ns(self) -> int:
        return _sec_to_nanosec(self.duration_s)

    @duration_ns.setter
    def duration_ns(self, value: int):
        self.duration_s = _nanosec_to_sec(value)

    @property
    def start_time_s(self) -> float:
        return _parse_time_string_to_seconds(time_str=self["Start Time"])

    @start_time_s.setter
    def start_time_s(self, value: float):
        self["Start Time"] = _format_seconds_to_time_string(sec=value)

    @property
    def start_time_ns(self) -> int:
        return _sec_to_nanosec(self.start_time_s)

    @start_time_ns.setter
    def start_time_ns(self, value: int):
        self.start_time_s = _nanosec_to_sec(value)

    @property
    def start_time_synced_s(self) -> float:
        return float(self["Start Time (Synced)"])

    @start_time_synced_s.setter
    def start_time_synced_s(self, value: float):
        self["Start Time (Synced)"] = value

    @property
    def start_time_synced_ns(self) -> int:
        return _sec_to_nanosec(self.start_time_synced_s)

    @start_time_synced_ns.setter
    def start_time_synced_ns(self, value: int):
        self.start_time_synced_s = _nanosec_to_sec(value)

    @property
    def world_camera_frames(self) -> int:
        return int(self["World Camera Frames"])

    @world_camera_frames.setter
    def world_camera_frames(self, value: int):
        self["World Camera Frames"] = int(value)

    @property
    def world_camera_resolution(self) -> T.Tuple[int, int]:
        resolution = self["World Camera Resolution"]
        resolution_match = re.search(r"^(\d+)x(\d+)$", resolution.strip())
        if not resolution_match:
            raise RecordingInfoInvalidError #TODO
        w, h = resolution_match[1], resolution_match[2]
        return int(w), int(h)

    @world_camera_resolution.setter
    def world_camera_resolution(self, value: T.Tuple[int, int]):
        w, h = value
        self["World Camera Resolution"] = f"{v}x{h}"

    @property
    def _schema(self):
        return {
            "Recording UUID": (_validate_recording_uuid, None),
            "Recording Name": (_validate_type(str), _default_recording_name),
            "Duration Time": (_validate_type(str), None), #TODO: Add better validation
            "Start Date": (_validate_type(str), None), #TODO: Add better validation
            "Start Time": (_validate_type(str), None), #TODO: Add better validation
            "Start Time (System)": (_validate_type(str), None), #TODO: Add better validation
            "Start Time (Synced)": (_validate_type(str), None), #TODO: Add better validation
            "Capture Software Version": (_validate_type(str), None),
            "Capture Software": (_validate_type(str), lambda info: info.CAPTURE_SOFTWARE_PUPIL_CAPTURE),
            "World Camera Frames": (_validate_type(str), lambda _: "0"), #TODO: Add better default
            "World Camera Resolution": (_validate_type(str), lambda _: "0x0"), #TODO: Add better validation and default
            "System Info": (_validate_type(str), lambda _: "..."), #TODO: Get default system information
        }

    # RecordingInfoFile

    @property
    def file_name(self) -> str:
        return "info.csv"

    @staticmethod
    def _read_dict_from_file(file) -> dict:
        return csv_utils.read_key_value_file(file)

    @staticmethod
    def _write_dict_to_file(file, dict_to_write: dict, sort_keys: bool):
        if sort_keys:
            ordered_dict = collections.OrderedDict()
            for key in sorted(dict_to_write.keys()):
                ordered_dict[key] = dict_to_write[key]
            dict_to_write = ordered_dict

        csv_utils.write_key_value_file(file, dict_to_write, append=False)

    # Public

    CAPTURE_SOFTWARE_PUPIL_CAPTURE = "Pupil Capture"

    @property
    def capture_software(self) -> str:
        return self.get("Capture Software", "Pupil Capture")

    @property
    def is_pupil_mobile(self) -> bool:
        return self.capture_software == "Pupil Mobile"

    @property
    def is_pupil_invisible(self) -> bool:
        return self.capture_software == "Pupil Invisible"


class RecordingInfoFileJSON(RecordingInfoFile):

    # RecordingInfo

    DEFAULT_ANDROID_DEVICE_ID = ""
    DEFAULT_ANDROID_DEVICE_NAME = ""
    DEFAULT_ANDROID_DEVICE_MODEL = ""

    @property
    def _schema(self):
        return {
            "data_format_version": (_validate_version, None),
            "recording_uuid": (_validate_recording_uuid, None),
            "app_version": (_validate_version, None),
            "start_time": (_validate_type(int), None),
            "duration": (_validate_type(int), None),
            "start_time_synced": (_validate_type(int), None),
            #TODO: Add all required keys...
            "android_device_id": (_validate_type(str), lambda info: info.DEFAULT_ANDROID_DEVICE_ID),
            "android_device_name": (_validate_type(str), lambda info: info.DEFAULT_ANDROID_DEVICE_NAME),
            "android_device_model": (_validate_type(str), lambda info: info.DEFAULT_ANDROID_DEVICE_MODEL),
            #TODO: Add all optional keys...
        }

    @property
    def recording_uuid(self) -> uuid.UUID:
        return uuid.UUID(self["recording_uuid"])

    @recording_uuid.setter
    def recording_uuid(self, value: uuid.UUID):
        self["recording_uuid"] = str(value)

    @property
    def recording_name(self) -> str:
        return self._get_cached_computed_property("_recording_name", _get_recording_name, rec_dir=self.rec_dir)

    @recording_name.setter
    def recording_name(self, value: str):
        self._set_cached_computed_property("_recording_name", value)

    @property
    def software_version(self) -> RecordingVersion:
        return RecordingVersion(self["app_version"])

    @software_version.setter
    def software_version(self, value: RecordingVersion):
        self["app_version"] = str(value) #TODO: Test if this conversion is correct

    @property
    def data_format_version(self) -> T.Optional[RecordingVersion]:
        return RecordingVersion(self["data_format_version"])

    @data_format_version.setter
    def data_format_version(self, value: T.Optional[RecordingVersion]):
        self["data_format_version"] = str(value)

    @property
    def duration_s(self) -> float:
        return _nanosec_to_sec(self.duration_ns)

    @duration_s.setter
    def duration_s(self, value: float):
        self.duration_ns = _sec_to_nanosec(value)

    @property
    def duration_ns(self) -> int:
        return int(self["duration"])

    @duration_ns.setter
    def duration_ns(self, value: int):
        self["duration"] = int(value)

    @property
    def start_time_s(self) -> float:
        return _nanosec_to_sec(self.start_time_ns)

    @start_time_s.setter
    def start_time_s(self, value: float):
        self.start_time_ns = _sec_to_nanosec(value)

    @property
    def start_time_ns(self) -> int:
        return int(self["start_time"])

    @start_time_ns.setter
    def start_time_ns(self, value: int):
        self["start_time"] = int(value)

    @property
    def start_time_synced_s(self) -> float:
        return _nanosec_to_sec(self.start_time_synced_ns)

    @start_time_synced_s.setter
    def start_time_synced_s(self, value: float):
        self.start_time_synced_ns = _sec_to_nanosec(value)

    @property
    def start_time_synced_ns(self) -> int:
        return int(self["start_time_synced"])

    @start_time_synced_ns.setter
    def start_time_synced_ns(self, value: int):
        self["start_time_synced"] = int(value)

    @property
    def world_camera_frames(self) -> int:
        frame_count = self._get_cached_computed_property(
            key="_world_camera_frames",
            getter=_get_world_camera_frame_count,
            rec_dir=self.rec_dir
        )
        return int(frame_count)

    @world_camera_frames.setter
    def world_camera_frames(self, value: int):
        self._set_cached_computed_property(
            key="_world_camera_frames",
            cached_value=int(value)
        )

    @property
    def world_camera_resolution(self) -> T.Tuple[int, int]:
        w, h = self._get_cached_computed_property(
            key="_world_camera_resolution",
            getter=_get_world_camera_resolution,
            rec_dir=self.rec_dir
        )
        return (int(w), int(h))

    @world_camera_resolution.setter
    def world_camera_resolution(self, value: T.Tuple[int, int]):
        w, h = value
        self._set_cached_computed_property(
            key="_world_camera_resolution",
            cached_value=(int(w), int(h))
        )

    # RecordingInfoFile

    def __init__(self, rec_dir: str, should_load_file: bool = True, should_validate: bool = True):
        self._info = {}
        self._cache = {}
        super().__init__(
            rec_dir=rec_dir,
            should_load_file=should_load_file,
            should_validate=should_validate,
        )

    @property
    def file_name(self) -> str:
        return "info.json"

    @staticmethod
    def _read_dict_from_file(file) -> dict:
        return json.load(file)

    @staticmethod
    def _write_dict_to_file(file, dict_to_write: dict, sort_keys: bool):
        json.dump(dict_to_write, file, indent=4, sort_keys=sort_keys)


### PRIVATE

def _nanosec_to_sec(ns: int) -> float:
    return float(ns / 1e9)

def _sec_to_nanosec(s: float) -> int:
    return int(s * 1e9)

def _get_recording_name(rec_dir: str) -> str:
    return os.path.basename(rec_dir)

def _get_world_camera_frame_count(rec_dir: str) -> int:
def _default_recording_name(info: RecordingInfoFile) -> str:
    return os.path.basename(info.rec_dir)


    return 0 #FIXME

def _get_world_camera_resolution(rec_dir: str) -> T.Tuple[int, int]:
    return (0, 0) #FIXME


def _parse_time_string_to_seconds(time_str: str, format="%H:%M:%S") -> float:
    t = time.strptime(time_str, format)
    t = datetime.timedelta(
        hours=t.tm_hour,
        minutes=t.tm_min,
        seconds=t.tm_sec,
    )
    return float(t.total_seconds())

def _format_seconds_to_time_string(sec: float) -> str:
    t = time.gmtime(sec)
    return time.strftime("%H:%M:%S", t)


def _validate_recording_uuid(input):
    try:
        _ = uuid.UUID(input)
        return None
    except Exception:
        return f"Invalid UUID string: {input}"


def _validate_version(input):
    if not isinstance(input, str):
        return f"Expected type: {str}, actual type: {type(input)}"
    #TODO: Validate version string format
    return None


def _validate_type(expected_type):
    def validation(input):
        if not isinstance(input, expected_type):
            return f"Expected type: {expected_type}, actual type: {type(input)}"
        return None
    return validation
