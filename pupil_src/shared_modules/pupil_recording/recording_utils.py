"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import enum
import os
from pathlib import Path

from packaging.version import Version
from typing_extensions import Literal

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
    NEON = enum.auto()
    CLOUD_CSV_EXPORT = enum.auto()


def get_recording_type(rec_dir: str) -> RecordingType:
    assert_valid_rec_dir(rec_dir)

    non_core_type = _is_PI_Neon_or_Cloud_export(rec_dir)

    if RecordingInfoFile.does_recording_contain_info_file(rec_dir):
        return RecordingType.NEW_STYLE

    elif _is_old_style_player_recording(rec_dir):
        return RecordingType.OLD_STYLE

    elif _is_pupil_mobile_recording(rec_dir):
        return RecordingType.MOBILE

    elif non_core_type:
        return non_core_type

    raise InvalidRecordingException(
        reason=f"There is no info file in the target directory.", recovery=""
    )


def assert_valid_rec_dir(rec_dir: str):
    """Checks if rec_dir is a directory.

    Raises InvalidRecordingException (with corresponding message) when:
        - rec_dir does not exist
        - rec_dir points to a video file instead of the directory
        - rec_dir points to a file in general
        - rec_dir is readable and writable
    """
    if not os.access(rec_dir, os.R_OK):
        raise InvalidRecordingException(
            reason=f"Player does not have sufficient permissions to read the directory",
            recovery="",
        )

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
            reason=f"Target at path does not exist:\n{rec_dir}", recovery=""
        )

    if not rec_dir.is_dir():
        if is_video_file(rec_dir):
            raise InvalidRecordingException(
                reason=f"The provided path is a video, not a recording directory",
                recovery="Please provide a recording directory.",
            )
        else:
            raise InvalidRecordingException(
                reason=f"Target at path is not a directory:\n{rec_dir}", recovery=""
            )

    if not os.access(rec_dir, os.W_OK):
        _attempt_changing_file_owners_on_macOS(rec_dir)

    if not os.access(rec_dir, os.W_OK):
        raise InvalidRecordingException(
            reason=f"Player must be able to write files to\n{rec_dir}",
            recovery="Please change the file permission accordingly.",
        )


# NOTE: The following functions are actually not correct given their name, only in the
# context of their usage above in get_recording_type(). They should not be called from
# outside!


def _is_PI_Neon_or_Cloud_export(
    rec_dir: str,
) -> Literal[
    RecordingType.INVISIBLE, RecordingType.NEON, RecordingType.CLOUD_CSV_EXPORT, None
]:
    if next(Path(rec_dir).glob("*.csv"), False):
        return RecordingType.CLOUD_CSV_EXPORT
    try:
        info = recording_info_utils.read_info_json_file(rec_dir)
        data_version = Version(info["data_format_version"])
        if data_version.major < 2:
            return RecordingType.INVISIBLE
        else:
            return RecordingType.NEON
    except (FileNotFoundError, KeyError):
        return None


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


def _attempt_changing_file_owners_on_macOS(rec_dir: str):
    """Attempts to change the ownership of the recording directory to the current user.

    This is a workaround for macOS permissions issues.
    """
    import platform

    if platform.system() != "Darwin":
        return

    import getpass
    import logging
    import subprocess
    import textwrap

    logger = logging.getLogger(__name__)
    user = getpass.getuser()
    ask_for_permissions_to_change_ownership = textwrap.dedent(
        f"""
        set theDialogText to "Pupil Player does not have sufficient file permissions to process this recording. If you proceed Player will change the file ownership to get the neccessary access."
        set continueText to "Proceed with administrator privileges"
        set cancelText to "Cancel"
        display dialog theDialogText buttons {{cancelText, continueText}} default button continueText cancel button cancelText
        do shell script "chown -R {user} '{rec_dir}'" with administrator privileges
        """
    )

    try:
        logger.debug(
            "Attempt to change file ownership using osascript:\n"
            f"{ask_for_permissions_to_change_ownership}"
        )
        subprocess.run(
            ["osascript", "-ss"],
            input=ask_for_permissions_to_change_ownership,
            encoding="utf-8",
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.exception(
            f"Attempt to change file ownership failed: {exc.stderr.strip()}"
        )
        raise InvalidRecordingException(
            reason=f"Player was not able to change the file ownership for\n{rec_dir}",
            recovery="Please change the file permissions manually.",
        ) from exc
