"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections.abc
import enum
import logging
import re
import typing as T
from pathlib import Path

from .info.recording_info import RecordingInfoFile, RecordingInfoInvalidError
from .recording_utils import (
    VALID_VIDEO_EXTENSIONS,
    InvalidRecordingException,
    assert_valid_rec_dir,
)

logger = logging.getLogger(__name__)


class PupilRecording:
    def __init__(self, rec_dir):
        self._info_file = None
        self.load(rec_dir=rec_dir)

    @property
    def meta_info(self) -> RecordingInfoFile:
        return self._info_file

    @property
    def rec_dir(self):
        return self.meta_info.rec_dir

    # Public

    def reload(self):
        self.load(rec_dir=self.rec_dir)

    def load(self, rec_dir):
        rec_dir = Path(rec_dir).resolve()

        assert_valid_rec_dir(rec_dir)

        try:
            info_file = RecordingInfoFile.read_file_from_recording(rec_dir=rec_dir)
        except FileNotFoundError:
            raise InvalidRecordingException(
                reason=(
                    f"There is no {RecordingInfoFile.file_name}"
                    " in the target directory"
                ),
                recovery="",
            )
        except RecordingInfoInvalidError as err:
            raise InvalidRecordingException(f"{err}")

        self._info_file = info_file

    class FileFilter(collections.abc.Sequence):
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
            ("audio"): r"^audio",
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
            ("mp4",): r"\.mp4$",
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
            return self.filter_multiple(
                "eye0", "eye1", mode=PupilRecording.FileFilter.FilterMode.UNION
            )

        def eye_id(self, eye_id: int) -> FilterType:
            return self.filter(f"eye{eye_id}")

        def videos(self) -> FilterType:
            return self.filter("videos")

        def audio(self) -> FilterType:
            return self.filter("audio")

        def mp4(self) -> FilterType:
            return self.filter("mp4")

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
                self.__files = list(set.union(*sets_of_files))
            elif mode is self.FilterMode.INTERSECTION:
                self.__files = list(set.intersection(*sets_of_files))
            else:
                logger.warning(
                    f"Unknown filter mode: {mode}! Must be 'union' or 'intersection'!"
                )
            return self

        def __init__(self, rec_dir: str):
            self.__files = sorted(filter(Path.is_file, Path(rec_dir).iterdir()))

        def __getitem__(self, key):
            # Used for implementing collections.abc.Sequence
            return self.__files[key]

        def __len__(self):
            # Used for implementing collections.abc.Sequence
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
