import collections
import enum
import logging
from pathlib import Path
import re
import typing as T
import uuid

import csv_utils
from video_capture.utils import VIDEO_EXTS as VALID_VIDEO_EXTENSIONS


from .recording_info import RecordingInfo, RecordingInfoInvalidError, RecordingInfoFileCSV, RecordingInfoFileJSON


logger = logging.getLogger(__name__)


class InvalidRecordingException(Exception):
    def __init__(self, reason: str, recovery: str = ""):
        message = (reason + "\n" + recovery) if recovery else reason
        super().__init__(message)
        self.reason = reason
        self.recovery = recovery

    def __str__(self):
        return f"{type(self).__name__}: {super().__str__()}"


class PupilRecording:
    def __init__(self, rec_dir):
        self._info_csv = None
        self.load(rec_dir=rec_dir)

    @property
    def meta_info(self) -> RecordingInfoFileCSV:
        return self._info_csv

    @property
    def rec_dir(self):
        return self.meta_info.rec_dir

    @property
    def capture_software(self) -> str:
        return self.meta_info.capture_software

    @property
    def data_format_version(self) -> T.Optional[str]:
        # TODO: self.meta_info.data_format_version returns a non-optional RecordingVersion instance;
        #       Investigate if that property is allowed to return None, and decide how to proceed with this API.
        return self.meta_info.data_format_version

    @property
    def is_pupil_mobile(self) -> bool:
        return self.capture_software == "Pupil Mobile"

    @property
    def is_pupil_invisible(self) -> bool:
        return self.capture_software == "Pupil Invisible"

    def reload(self):
        self.load(rec_dir=self.rec_dir)

    def load(self, rec_dir):
        rec_dir = Path(rec_dir).resolve()

        def normalize_extension(ext: str) -> str:
            if ext.startswith("."):
                ext = ext[1:]
            return ext

        def is_video_file(file_path):
            if not file_path.is_file():
                return False
            ext = file_path.suffix
            ext = normalize_extension(ext)
            valid_video_extensions = map(normalize_extension, VALID_VIDEO_EXTENSIONS)
            if ext not in valid_video_extensions:
                return False
            return True

        if not rec_dir.exists():
            raise InvalidRecordingException(
                reason=f"Target at path does not exist: {rec_dir}", recovery=""
            )

        if not rec_dir.is_dir():
            if is_video_file(rec_dir):
                raise InvalidRecordingException(
                    reason=f"The provided path is a video, not a recording directory",
                    recovery="Please provide a recording directory",
                )
            else:
                raise InvalidRecordingException(
                    reason=f"Target at path is not a directory: {rec_dir}", recovery=""
                )

        try:
            # Load the info.csv file without validating it, since the data might be split between info.csv and info.json
            info_csv = RecordingInfoFileCSV(rec_dir=rec_dir, should_validate=False)
        except FileNotFoundError:
            raise InvalidRecordingException(
                reason=f"There is no info.csv in the target directory",
                recovery=""
            )

        info_json = None

        if info_csv.is_pupil_invisible:
            try:
                info_json = RecordingInfoFileJSON(rec_dir=rec_dir, should_validate=False)
            except (FileNotFoundError, RecordingInfoInvalidError) as err:
                pass
            else:

                try:
                    _ = info_json.recording_uuid
                except KeyError:
                    info_json.recording_uuid = uuid.uuid4()

                # Overwrite info.csv with data from info.json
                info_csv.copy_from(info_json)
                info_csv.capture_software = RecordingInfoFileCSV.CAPTURE_SOFTWARE_PUPIL_INVISIBLE
                info_csv.data_format_version = None
                info_csv.save_file()

        try:
            # At this point the info.csv file must be valid
            info_csv.validate()
        except RecordingInfoInvalidError as err:
            raise InvalidRecordingException(
                reason=f"{err}",
                recovery=""
            )
        else:
            if info_json is not None:
                info_json.remove_file()
                del info_json

        all_file_paths = rec_dir.glob("*")

        # TODO: Should this validation be "are there any video files" or are there specific video files?
        if not any(is_video_file(path) for path in all_file_paths):
            raise InvalidRecordingException(
                reason=f"Target directory does not contain any video files", recovery=""
            )

        # TODO: Are there any other validations missing?
        # All validations passed

        self._info_csv = info_csv

    class FileFilter(collections.Sequence):
        """Utility class for conveniently filtering files of the recording.

        Filters can be applied sequentially, since they return a filter again.
        Overloading __getitem__ and __len__ allows for full sequence functionality.
        Example usage:

            # prints all world videos in my_rec_dir
            for path in FileFilter("my_rec_dir").videos().world():
                print(f"World video file: {path}")
        """

        FilterType = "PupilRecording.FileFilter"

        PATTERNS = {
            ("core", "world"): r"^world",
            ("core", "eye0"): r"^eye0",
            ("core", "eye1"): r"^eye1",
            ("mobile", "world"): [
                r"^Pupil Cam(\d) ID2",  # pupil core headset
                r"^Logitech Webcam C930e",  # old headset with logitech webcam
            ],
            ("mobile", "eye0"): r"^Pupil Cam(\d) ID0",
            ("mobile", "eye1"): r"^Pupil Cam(\d) ID1",
            ("pi", "world"): r"^PI world v(\d+) ps(\d+)",
            ("pi", "eye0"): r"^PI left v(\d+) ps(\d+)",
            ("pi", "eye1"): r"^PI right v(\d+) ps(\d+)",
            ("videos",): [rf"\.{ext}$" for ext in VALID_VIDEO_EXTENSIONS],
            ("rawtimes",): r"\.time$",
            ("timestamps",): r"_timestamps\.npy$",
            ("lookup",): r"_lookup\.npy$",
        }

        def world(self) -> FilterType:
            return self.filter("world")

        def eye0(self) -> FilterType:
            return self.filter("eye0")

        def eye1(self) -> FilterType:
            return self.filter("eye1")

        def eyes(self) -> FilterType:
            return self.filter_mutliple("eye0", "eye1", mode="union")

        def videos(self) -> FilterType:
            return self.filter("videos")

        def raw_time(self) -> FilterType:
            return self.filter("rawtimes")

        def timestamps(self) -> FilterType:
            return self.filter("timestamps")

        def lookup(self) -> FilterType:
            return self.filter("lookup")

        def core(self) -> FilterType:
            return self.filter("core")

        def mobile(self) -> FilterType:
            return self.filter("mobile")

        def pi(self) -> FilterType:
            return self.filter("pi")

        def filter(self, key: str) -> FilterType:
            """Filters files by key from the PATTERNS dict.

            Keeps all files that match any pattern which contains the key.
            """
            return self.filter_patterns(*self.patterns_with_key(key))

        @enum.unique
        class FilterMode(enum.Enum):
            UNION = 1
            INTERSECTION = 2

        def filter_multiple(self, *keys: str, mode: FilterMode) -> FilterType:
            """Filters files by multiple keys from the PATTERNS dict.

            Mode determines aggregation of resulting files for every key.
            """
            patterns_for_keys = [self.patterns_with_key(key) for key in keys]
            sets_of_files = [
                set(self.files_with_patterns(*patterns))
                for patterns in patterns_for_keys
            ]
            if mode is self.FilterMode.UNION:
                self.__files = set.union(*sets_of_files)
            elif mode is self.FilterMode.INTERSECTION:
                self.__files = set.intersection(*sets_of_files)
            else:
                logger.warning(
                    f"Unknown filter mode: {mode}! Must be 'union' or 'intersection'!"
                )
            return self

        def __init__(self, rec_dir: str):
            self.__files = [path for path in Path(rec_dir).iterdir() if path.is_file()]

        def __getitem__(self, key):
            # Used for implementing collections.Sequence
            return self.__files[key]

        def __len__(self):
            # Used for implementing collections.Sequence
            return len(self.__files)

        def filter_patterns(self, *patterns: str) -> FilterType:
            """Filters current files, keeping anything matching any of the patterns."""
            self.__files = self.files_with_patterns(*patterns)
            return self

        def files_with_patterns(self, *patterns: str) -> T.Sequence[Path]:
            return [
                item
                for item in self.__files
                if any([re.search(pattern, item.name) for pattern in patterns])
            ]

        @classmethod
        def patterns_with_key(cls, key: str):
            for keys, pattern in cls.PATTERNS.items():
                if key in keys:
                    if isinstance(pattern, list):
                        yield from pattern
                    else:
                        yield pattern

    def files(self) -> "PupilRecording.FileFilter":
        return PupilRecording.FileFilter(self.rec_dir)

