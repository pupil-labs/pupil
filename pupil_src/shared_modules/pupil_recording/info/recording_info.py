"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import collections
import json
import logging
import math
import os
import typing as T
import uuid

from version_utils import ParsedVersion, get_version, parse_version

__all__ = ["RecordingInfo", "RecordingInfoFile", "RecordingInfoInvalidError"]


logger = logging.getLogger(__name__)


class RecordingInfoInvalidError(Exception):
    @staticmethod
    def missingKey(key: str) -> "RecordingInfoInvalidError":
        return RecordingInfoInvalidError(f'Key "{key}" is missing')

    @staticmethod
    def wrongTypeForKey(
        key: str, actual_type, expected_type
    ) -> "RecordingInfoInvalidError":
        return RecordingInfoInvalidError(
            f'Value for key "{key}" is of the wrong type "{actual_type}"; '
            f'expected "{expected_type}"'
        )


class RecordingInfo(collections.abc.MutableMapping):

    # MutableMapping

    def __getitem__(self, key):
        return self.__storage.__getitem__(key)

    def __setitem__(self, key, item):
        return self.__storage.__setitem__(key, item)

    def __delitem__(self, key):
        return self.__storage.__delitem__(key)

    def __iter__(self):
        return self.__storage.__iter__()

    def __len__(self):
        return self.__storage.__len__()

    # Public

    RECORDING_SOFTWARE_NAME_PUPIL_CAPTURE = "Pupil Capture"
    RECORDING_SOFTWARE_NAME_PUPIL_MOBILE = "Pupil Mobile"
    RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE = "Pupil Invisible"

    @property
    @abc.abstractmethod
    def meta_version(self) -> ParsedVersion:
        pass

    @property
    @abc.abstractmethod
    def min_player_version(self) -> ParsedVersion:
        pass

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
    def start_time_system_s(self) -> float:
        pass

    @start_time_system_s.setter
    @abc.abstractmethod
    def start_time_system_s(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def start_time_system_ns(self) -> int:
        pass

    @start_time_system_ns.setter
    @abc.abstractmethod
    def start_time_system_ns(self, value: int):
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
    def recording_software_name(self) -> str:
        pass

    @recording_software_name.setter
    @abc.abstractmethod
    def recording_software_name(self, value: str):
        pass

    @property
    @abc.abstractmethod
    def recording_software_version(self) -> str:
        pass

    @recording_software_version.setter
    @abc.abstractmethod
    def recording_software_version(self, value: str):
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
    def system_info(self) -> str:
        pass

    @system_info.setter
    @abc.abstractmethod
    def system_info(self, value: str):
        pass

    def validate(self):
        try:
            self._validate_public_interface()
        except RecordingInfoInvalidError:
            raise
        except Exception as err:
            raise RecordingInfoInvalidError(f"Validation failed with exception: {err}")

    @classmethod
    def property_equality(cls, x: "RecordingInfo", y: "RecordingInfo") -> bool:
        try:
            cls._assert_property_equality(x, y)
            return True
        except AssertionError:
            return False

    def update_writeable_properties_from(self, other):
        dest, src = self, other

        for (
            property_name,
            ((_, dest_setter), (src_getter, _)),
        ) in self.__matching_public_properties(dest, src).items():
            if dest_setter is None:
                # readonly property
                continue
            value = src_getter(src)
            dest_setter(dest, value)

    @classmethod
    def _assert_property_equality(cls, x: "RecordingInfo", y: "RecordingInfo") -> bool:
        def equal_seconds(x: float, y: float) -> bool:
            # Decimal precision is lost when writing to / reading from info.csv
            return math.floor(x) == math.floor(y)

        for (
            property_name,
            ((x_getter, _), (y_getter, _)),
        ) in cls.__matching_public_properties(x, y).items():
            x_value = x_getter(x)
            y_value = y_getter(y)

            if (
                property_name.endswith("_s")
                and isinstance(x_value, float)
                and isinstance(y_value, float)
            ):
                # Compare seconds using `equal_seconds` instead of `==`
                assert equal_seconds(x_value, y_value), f"TODO"
            else:
                assert x_value == y_value, f"TODO"

    # Internal

    _PublicGetter = T.Callable[["RecordingInfo"], T.Any]
    _PublicSetter = T.Callable[["RecordingInfo", T.Any], None]
    _PublicProperty = T.Tuple[_PublicGetter, _PublicSetter]

    @property
    def _public_properties(self) -> T.Mapping[str, _PublicProperty]:
        return {
            "meta_version": (type(self).meta_version.fget, None),
            "min_player_version": (type(self).min_player_version.fget, None),
            "recording_uuid": (
                type(self).recording_uuid.fget,
                type(self).recording_uuid.fset,
            ),
            "start_time_system_s": (
                type(self).start_time_system_s.fget,
                type(self).start_time_system_s.fset,
            ),
            "start_time_synced_s": (
                type(self).start_time_synced_s.fget,
                type(self).start_time_synced_s.fset,
            ),
            "duration_s": (type(self).duration_s.fget, type(self).duration_s.fset),
            "recording_software_name": (
                type(self).recording_software_name.fget,
                type(self).recording_software_name.fset,
            ),
            "recording_software_version": (
                type(self).recording_software_version.fget,
                type(self).recording_software_version.fset,
            ),
            "recording_name": (
                type(self).recording_name.fget,
                type(self).recording_name.fset,
            ),
            "system_info": (type(self).system_info.fget, type(self).system_info.fset),
        }

    def _validate_public_interface(self):
        for property_name, (getter, setter) in self._public_properties.items():
            try:
                getter(self)
            except Exception as err:
                RecordingInfoInvalidError(
                    f'Accessing property "{property_name}" failed with exception: {err}'
                )

    # Private

    @property
    def __storage(self) -> dict:
        try:
            return self.__private_storage
        except AttributeError:
            self.__private_storage = {}
            return self.__private_storage

    @classmethod
    def __matching_public_properties(
        cls, x: "RecordingInfo", y: "RecordingInfo"
    ) -> T.Mapping[str, T.Tuple[_PublicProperty, _PublicProperty]]:

        x_properties = x._public_properties
        y_properties = y._public_properties

        x_property_names = set(x_properties.keys())
        y_property_names = set(y_properties.keys())

        if x_property_names == y_property_names:
            property_names = x_property_names
        else:
            property_names = x_property_names.intersection(y_property_names)
            logger.debug(
                "Public property mismatch; will only check the following properties: "
                f"{property_names}"
            )

        return {key: (x_properties[key], y_properties[key]) for key in property_names}


class RecordingInfoFile(RecordingInfo):

    # Public

    file_name = "info.player.json"

    def __init__(self, rec_dir: str, should_load_file: bool, should_validate: bool):
        self._rec_dir = str(rec_dir)
        if should_load_file:
            self.load_file(should_validate=should_validate)

    @property
    def rec_dir(self) -> str:
        return self._rec_dir

    @property
    def file_path(self) -> str:
        return RecordingInfoFile._info_file_path(self.rec_dir)

    @property
    def does_file_exist(self) -> bool:
        return os.path.isfile(self.file_path)

    def save_file(self, should_validate: bool = True, sort_keys: bool = True):
        if should_validate:
            self.validate()
        with open(self.file_path, "w") as file:
            self._write_dict_to_file(
                file=file, dict_to_write=self._dict_to_save(), sort_keys=sort_keys
            )

    def load_file(self, should_validate: bool = True):
        """
        Load the data from the info file in the recording directory.
        :param should_validate: If `True`, validates the loaded data.
        """
        with open(self.file_path) as file:
            read_dict = self._read_dict_from_file(file=file)
        self.update(read_dict)
        if should_validate:
            self.validate()

    def remove_file(self):
        try:
            os.remove(self.file_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def does_recording_contain_info_file(rec_dir: str) -> bool:
        file_path = RecordingInfoFile._info_file_path(rec_dir=rec_dir)
        return os.path.isfile(file_path)

    @staticmethod
    def detect_recording_info_file_version(rec_dir: str) -> ParsedVersion:
        file_path = RecordingInfoFile._info_file_path(rec_dir=rec_dir)
        with open(file_path) as file:
            read_dict = RecordingInfoFile._read_dict_from_file(file=file)
        return parse_version(read_dict["meta_version"])

    @staticmethod
    def read_file_from_recording(rec_dir: str) -> "RecordingInfoFile":
        info_file_version = RecordingInfoFile.detect_recording_info_file_version(
            rec_dir
        )
        try:
            info_file_class = RecordingInfoFile._info_file_versions[info_file_version]

        except KeyError:
            latest_version = RecordingInfoFile.get_latest_info_file_version()
            if info_file_version <= latest_version:
                # Something went really wrong with our versioning! Should never happen!
                raise RecordingInfoInvalidError("Unexpected version order error!")

            # We don't have a template for this meta version. Check for
            # min_player_version and try to find a best template.
            try:
                info_file_path = RecordingInfoFile._info_file_path(rec_dir)
                with open(info_file_path) as f:
                    info_dict = RecordingInfoFile._read_dict_from_file(f)
                min_player_version = parse_version(info_dict["min_player_version"])
            except Exception as e:
                # Catching BaseException since at this point we don't know anything
                logger.error(
                    f"Exception during trying to load min_player_version for recording"
                    f" with meta version {info_file_version} from player with latest"
                    f" meta version {latest_version}: {str(e)}"
                )
                raise RecordingInfoInvalidError(
                    f"Recording is too new to be opened with this version of Player!"
                )

            if min_player_version > get_version():
                raise RecordingInfoInvalidError(
                    f"This recording requires Player version >= {min_player_version}!"
                )

            # At this point we should be safe, but warn the user anyways
            logger.warning(
                "Opening recording of newer version than this version of Pupil Player."
                " This might lead to problems."
                " Please consider updating to the latest version of Pupil Player!"
            )
            logger.debug(
                f"Trying to open info file meta version {info_file_version}"
                f" with template for version {latest_version}!"
            )
            info_file_class = RecordingInfoFile._info_file_versions[latest_version]

        return info_file_class(
            rec_dir=rec_dir, should_load_file=True, should_validate=True
        )

    @staticmethod
    def create_empty_file(
        rec_dir: str, fixed_version: T.Optional[ParsedVersion] = None
    ) -> "RecordingInfoFile":
        """
        Creates a new `RecordingInfoFile` instance using the latest meta format version,
        but without any data.

        :param rec_dir: Path to the recording directory.
        """
        if fixed_version is None:
            # use latest version as default
            fixed_version = max(RecordingInfoFile._info_file_versions)
        info_file_class = RecordingInfoFile._info_file_versions[fixed_version]
        return info_file_class(
            rec_dir=rec_dir, should_load_file=False, should_validate=False
        )

    def validate(self):
        for key, (validator, default_getter) in self._private_key_schema.items():
            if key not in self:
                if default_getter is None:
                    raise RecordingInfoInvalidError(f'Missing required key: "{key}"')
                else:
                    self[key] = default_getter(self)
                    continue
            try:
                validator(self[key])
            except Exception as err:
                raise RecordingInfoInvalidError(
                    f"Validation failed with exception: {err}"
                )
        super().validate()

    # Internal

    _KeyValueValidator = T.Callable[[T.Any], None]
    _KeyValueDefaultGetter = T.Callable[["RecordingInfoFile"], T.Any]
    _KeyValueSchema = T.Mapping[
        str, T.Tuple[_KeyValueValidator, T.Optional[_KeyValueDefaultGetter]]
    ]

    _info_file_versions = {}

    @classmethod
    def register_child_class(cls, version: ParsedVersion, child_class: type):
        """Use this to register interface implementations for specific versions."""
        # NOTE: This is dependency inversion to avoids circular imports, because we
        # don't need to know our child classes.
        # TODO: Would be much cleaner with self-registering meta classes.
        cls._info_file_versions[version] = child_class

    @classmethod
    def get_latest_info_file_version(cls) -> ParsedVersion:
        if not cls._info_file_versions:
            raise ValueError(
                "RecordingInfoFile not correctly initialized! No templates registered."
            )
        return sorted(cls._info_file_versions.keys())[-1]

    @property
    @abc.abstractmethod
    def _private_key_schema(self) -> _KeyValueSchema:
        """
        Schema of the data representation used when reading from / writing to the file.
        """
        pass

    @staticmethod
    def _info_file_path(rec_dir: str) -> str:
        """
        :param rec_dir: Path to the recording directory.
        :return: Full path of the file.
        """
        return os.path.join(rec_dir, RecordingInfoFile.file_name)

    @staticmethod
    def _read_dict_from_file(file) -> dict:
        """
        Read and deserialized data from a file handle.
        :param file: File handle to read from.
        :return: Deserialized data.
        """
        return json.load(file)

    @staticmethod
    def _write_dict_to_file(file, dict_to_write: dict, sort_keys: bool):
        json.dump(dict_to_write, file, indent=4, sort_keys=sort_keys)

    def _keys_to_save(self) -> T.Set[str]:
        """
        :return: Set of keys that will present in the saved file.
        """
        return set(self._private_key_schema.keys())

    def _dict_to_save(self):
        """
        Generates a new `dict` that should be serialized and saved in the file.
        """
        dict_to_save = {}
        for key in self._keys_to_save():
            try:
                dict_to_save[key] = self[key]
            except KeyError:
                raise RecordingInfoInvalidError.missingKey(key=key)
        return dict_to_save
