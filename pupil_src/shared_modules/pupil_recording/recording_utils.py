"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import enum
from pathlib import Path

from .info import recording_info_utils
from .info.recording_info import RecordingInfoFile

VALID_VIDEO_EXTENSIONS = ("mp4", "mjpeg", "h264", "mkv", "avi", "fake")


class InvalidRecordingException(Exception):
    def __init__(self, reason: str, recovery: str = ""):
        message = (reason + "\n" + recovery) if recovery else reason
        super().__init__(message)
        self.reason = reason
        self.recovery = recovery

    def __str__(self):
        return f"{type(self).__name__}: {super().__str__()}"


def assert_valid_recording_type(rec_dir: str):
    """Checks if rec_dir is (any) valid pupil recording.

    This does not assure that it can be opened immediately by player!
    Valid types are also old_style, mobile and invisible, which have to upgraded first.

    Raises InvalidRecordingException with a message if rec_dir is not valid.
    """
    assert get_recording_type(rec_dir) in RecordingType


class RecordingType(enum.Enum):
    MOBILE = enum.auto()
    INVISIBLE = enum.auto()
    OLD_STYLE = enum.auto()
    NEW_STYLE = enum.auto()


def get_recording_type(rec_dir: str) -> RecordingType:
    assert_valid_rec_dir(rec_dir)

    if RecordingInfoFile.does_recording_contain_info_file(rec_dir):
        return RecordingType.NEW_STYLE

    elif _is_old_style_player_recording(rec_dir):
        return RecordingType.OLD_STYLE

    elif _is_pupil_invisible_recording(rec_dir):
        return RecordingType.INVISIBLE

    elif _is_pupil_mobile_recording(rec_dir):
        return RecordingType.MOBILE

    raise InvalidRecordingException(
        reason=f"There is no info file in the target directory.", recovery=""
    )


def assert_valid_rec_dir(rec_dir: str):
    """Checks if rec_dir is a directory.

    Raises InvalidRecordingException (with corresponding message) when:
        - rec_dir does not exist
        - rec_dir points to a video file instead of the directory
        - rec_dir points to a file in general
    """
    rec_dir = Path(rec_dir).resolve()

    def normalize_extension(ext: str) -> str:
        if ext.startswith("."):
            ext = ext[1:]
        return ext

    def is_video_file(file_path: Path):
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


# NOTE: The following functions are actually not correct given their name, only in the
# context of their usage above in get_recording_type(). They should not be called from
# outside!


def _is_pupil_invisible_recording(rec_dir: str) -> bool:
    try:
        recording_info_utils.read_info_json_file(rec_dir)
        return True
    except FileNotFoundError:
        return False


def _is_pupil_mobile_recording(rec_dir: str) -> bool:
    try:
        info_csv = recording_info_utils.read_info_csv_file(rec_dir)
        return (
            info_csv["Capture Software"] == "Pupil Mobile"
            and "Data Format Version" not in info_csv
        )
    except (KeyError, FileNotFoundError):
        return False


def _is_old_style_player_recording(rec_dir: str) -> bool:
    """Return true if this is an old-style recording.

    There's two cases where we have old-style recordings:

        1.  The recording has been made with Pupil Capture < v1.16
            (and never been upgraded to new-style)
        2.  The recording has been made with Pupil Mobile and openend
            with Pupil Player < v1.16
    """
    try:
        info_csv = recording_info_utils.read_info_csv_file(rec_dir)
    except FileNotFoundError:
        return False

    # NOTE:
    # 1. "Unopened" Pupil Mobile recordings do not have a "Data Format Version" field.
    # 2. Very old versions of Pupil Capture also did not write "Data Format Version".
    is_newer_old_style = "Data Format Version" in info_csv
    is_not_pupil_mobile = info_csv.get("Capture Software", "") != "Pupil Mobile"
    return is_newer_old_style or is_not_pupil_mobile
