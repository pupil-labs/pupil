"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import re
import uuid
from pathlib import Path

from version_utils import parse_version

from ..info import RecordingInfoFile
from ..info import recording_info_utils as utils
from ..recording import PupilRecording
from ..recording_utils import InvalidRecordingException
from . import update_utils

# NOTE: Due to Pupil Mobile not having a data format version, we are using the software
# version here. The idea is to use Major.Minor specifically. This means that the
# software version of Pupil Mobile should can be increased in the patch version part
# only if this won't need additional update methods here.
NEXT_UNSUPPORTED_VERSION = parse_version("1.3")

logger = logging.getLogger(__name__)


def transform_mobile_to_corresponding_new_style(rec_dir: str) -> RecordingInfoFile:
    logger.info("Transform Pupil Mobile to new style recording...")
    info_csv = utils.read_info_csv_file(rec_dir)

    mobile_version = parse_version(info_csv["Capture Software Version"])

    if mobile_version >= NEXT_UNSUPPORTED_VERSION:
        raise InvalidRecordingException(
            "This Player version does not support Pupil Mobile versions >= "
            f"{NEXT_UNSUPPORTED_VERSION}. Got {mobile_version}."
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

    # Get information about recording from info.csv
    try:
        recording_uuid = info_csv.get("Recording UUID", uuid.uuid4())
        start_time_system_s = float(info_csv["Start Time (System)"])
        start_time_synced_s = float(info_csv["Start Time (Synced)"])
        duration_s = utils.parse_duration_string(info_csv["Duration Time"])
        recording_software_name = info_csv["Capture Software"]
        recording_software_version = info_csv["Capture Software Version"]
        recording_name = info_csv.get(
            "Recording Name", utils.default_recording_name(rec_dir)
        )
        system_info = info_csv.get("System Info", utils.default_system_info(rec_dir))
    except KeyError as e:
        logger.debug(f"KeyError while parsing mobile info.csv: {str(e)}")
        raise InvalidRecordingException(
            "This recording needs a data format update.\n"
            "Open it once in Pupil Player v1.17 to perform the update."
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
