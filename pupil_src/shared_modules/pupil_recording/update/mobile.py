"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import datetime
import logging
import re
import uuid
from pathlib import Path

import numpy as np

from .. import Version
from ..info import RecordingInfoFile
from ..info import recording_info_utils as utils
from ..recording import PupilRecording
from ..recording_utils import InvalidRecordingException
from . import update_utils

# NOTE: Due to Pupil Mobile not having a data format version, we are using the software
# version here. The idea is to use Major.Minor specifically. This means that the
# software version of Pupil Mobile should can be increased in the patch version part
# only if this won't need additional update methods here.
NEXT_UNSUPPORTED_VERSION = Version("1.3")

logger = logging.getLogger(__name__)


def transform_mobile_to_corresponding_new_style(rec_dir: str) -> RecordingInfoFile:
    logger.info("Transform Pupil Mobile to new style recording...")
    info_csv = utils.read_info_csv_file(rec_dir)

    mobile_version = Version(info_csv["Capture Software Version"])

    if mobile_version >= NEXT_UNSUPPORTED_VERSION:
        raise InvalidRecordingException(
            (
                "This Player version does not support Pupil Mobile versions >= "
                f"{NEXT_UNSUPPORTED_VERSION}. Got {mobile_version}."
            )
        )

    # elif mobile_version >= 3.0:
    #     ...
    # elif mobile_version >= 2.0:
    #     ...

    else:
        _transform_mobile_v1_2_to_pprf_2_0(rec_dir)


def _transform_mobile_v1_2_to_pprf_2_0(rec_dir: str):
    _generate_pprf_2_0_info_file(rec_dir)

    # rename info.csv file to info.mobile.csv
    info_csv = Path(rec_dir) / "info.csv"
    new_path = info_csv.with_name("info.mobile.csv")
    info_csv.replace(new_path)

    recording = PupilRecording(rec_dir)

    # patch world.intrinsics
    # NOTE: could still be worldless at this point
    update_utils._try_patch_world_instrinsics_file(
        rec_dir, recording.files().mobile().world().videos()
    )

    _rename_mobile_files(recording)
    _rewrite_timestamps(recording)


def _generate_pprf_2_0_info_file(rec_dir: str) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)

    try:
        # Get information about recording from info.csv
        recording_uuid = info_csv.get("Recording UUID", uuid.uuid4())

        # Allow inference of missing values in v1.16
        # TODO: Remove value inference in v1.17
        start_time_system_s = float(
            info_csv.get(
                "Start Time (System)", _infer_start_time_system_from_legacy(info_csv)
            )
        )
        start_time_synced_s = float(
            info_csv.get(
                "Start Time (Synced)", _infer_start_time_synced_from_legacy(rec_dir)
            )
        )
        duration_s = utils.parse_duration_string(info_csv["Duration Time"])
        recording_software_name = info_csv["Capture Software"]
        recording_software_version = info_csv["Capture Software Version"]
        recording_name = info_csv.get(
            "Recording Name", utils.default_recording_name(rec_dir)
        )
        system_info = info_csv.get("System Info", utils.default_system_info(rec_dir))
    except KeyError as e:
        logger.debug(f"KeyError while parsing old-style info.csv: {str(e)}")
        raise InvalidRecordingException(
            "This recording is too old to be opened with this version of Player!"
        )

    # Create a recording info file with the new format, fill out
    # the information, validate, and return.
    new_info_file = RecordingInfoFile.create_empty_file(rec_dir)
    new_info_file.recording_uuid = recording_uuid
    new_info_file.start_time_system_s = start_time_system_s
    new_info_file.start_time_synced_s = start_time_synced_s
    new_info_file.duration_s = duration_s
    new_info_file.recording_software_name = recording_software_name
    new_info_file.recording_software_version = recording_software_version
    new_info_file.recording_name = recording_name
    new_info_file.system_info = system_info
    new_info_file.validate()
    new_info_file.save_file()


def _rename_mobile_files(recording: PupilRecording):
    for path in recording.files():
        # replace prefix based on cam_id, part number is already correct
        pupil_cam = r"Pupil Cam\d ID(?P<cam_id>\d)"
        logitech_cam = r"Logitech Webcam C930e"
        prefixes = rf"({pupil_cam}|{logitech_cam})"
        match = re.match(rf"^(?P<prefix>{prefixes})", path.name)
        if match:
            replacement_for_cam_id = {
                "0": "eye0",
                "1": "eye1",
                "2": "world",
                None: "world",
            }
            replacement = replacement_for_cam_id[match.group("cam_id")]
            new_name = path.name.replace(match.group("prefix"), replacement)
            path.replace(path.with_name(new_name))  # rename with overwrite


def _rewrite_timestamps(recording: PupilRecording):
    update_utils._rewrite_times(recording, dtype=">f8")


def _infer_start_time_system_from_legacy(info_csv):
    _warn_imprecise_value_inference()
    logger.warning(f"Missing meta info key: `Start Time (System)`.")

    # Read date and time from info_csv
    string_start_date = info_csv["Start Date"]
    string_start_time = info_csv["Start Time"]

    # Combine and parse to datetime.datetime
    string_start_date_time = f"{string_start_date} {string_start_time}"
    format_date_time = "%d:%m:%Y %H:%M:%S"
    try:
        date_time = datetime.datetime.strptime(string_start_date_time, format_date_time)
    except ValueError as valerr:
        raise InvalidRecordingException(
            "Could not infer missing `Start Time (System)` value.\nUnexpected date time"
            f" input format: {string_start_date_time}"
        ) from valerr
    # Convert to Unix timestamp
    ts_start_date_time = date_time.timestamp()

    logger.info(f"Using {date_time} as input for `Start Time (System)` inference.")
    logger.info(f"Inferred `Start Time (System)`: {ts_start_date_time}")

    return ts_start_date_time


def _infer_start_time_synced_from_legacy(rec_dir):
    _warn_imprecise_value_inference()
    logger.warning(f"Missing meta info key: `Start Time (Synced)`.")

    files = PupilRecording.FileFilter(rec_dir)
    raw_time_files = files.mobile().raw_time()
    first_ts_per_raw_time_file = []
    for raw_time_file in raw_time_files:
        raw_time = np.fromfile(str(raw_time_file), dtype=">f8")
        if raw_time.size == 0:
            continue
        first_ts_per_raw_time_file.append(raw_time[0])
    if not first_ts_per_raw_time_file:
        raise InvalidRecordingException(
            "Could not infer missing `Start Time (Synced)` value. No timestamps found."
        )
    inferred_start_time_synced = min(first_ts_per_raw_time_file)
    logger.info(f"Inferred `Start Time (Synced)`: {inferred_start_time_synced}")
    return inferred_start_time_synced


# global variable to warn only once
_SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE = True


def _warn_imprecise_value_inference():
    global _SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE
    if not _SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE:
        return
    logger.warning(
        "\n\n!! Deprecation Warning !! Pupil Mobile recordings recorded with older"
        " versions than r0.21.0 are deprecated and will not be supported by future"
        " Pupil Player versions!\n"
    )
    logger.warning(
        "\n\n!! Imprecise Value Inference !! In order to upgrade a deprecated"
        " recording, Pupil Player needs to infer missing meta data from the existing"
        " recording. This inference is imprecise and might cause issues when converting"
        " recorded Pupil time to wall clock time.\n"
    )
    _SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE = False
