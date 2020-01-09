"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import re
from pathlib import Path

import numpy as np

import file_methods as fm
import methods as m
from video_capture.utils import pi_gaze_items

from .. import Version
from ..info import RecordingInfoFile
from ..info import recording_info_utils as utils
from ..recording import PupilRecording
from ..recording_utils import InvalidRecordingException
from . import update_utils

logger = logging.getLogger(__name__)

NEWEST_SUPPORTED_VERSION = Version("1.2")


def transform_invisible_to_corresponding_new_style(rec_dir: str):
    logger.info("Transform Pupil Invisible to new style recording...")
    info_json = utils.read_info_json_file(rec_dir)
    pi_version = Version(info_json["data_format_version"])

    if pi_version > NEWEST_SUPPORTED_VERSION:
        raise InvalidRecordingException(
            f"This version of player is too old! Please upgrade."
        )

    # elif pi_version > 3.0:
    #     ...
    # elif pi_version > 2.0:
    #     ...

    else:
        _transform_invisible_v1_0_to_pprf_2_1(rec_dir)


def _transform_invisible_v1_0_to_pprf_2_1(rec_dir: str):
    _generate_pprf_2_1_info_file(rec_dir)

    # rename info.json file to info.invisible.json
    info_json = Path(rec_dir) / "info.json"
    new_path = info_json.with_name("info.invisible.json")
    info_json.replace(new_path)

    recording = PupilRecording(rec_dir)

    # patch world.intrinsics
    # NOTE: could still be worldless at this point
    update_utils._try_patch_world_instrinsics_file(
        rec_dir, recording.files().pi().world().videos()
    )

    _rename_pi_files(recording)
    _rewrite_timestamps(recording)
    _convert_gaze(recording)


def _generate_pprf_2_1_info_file(rec_dir: str) -> RecordingInfoFile:
    info_json = utils.read_info_json_file(rec_dir)

    # Get information about recording from info.csv and info.json
    recording_uuid = info_json["recording_id"]
    start_time_system_ns = int(info_json["start_time"])
    start_time_synced_ns = int(info_json["start_time"])
    duration_ns = int(info_json["duration"])
    recording_software_name = RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
    recording_software_version = info_json["app_version"]
    recording_name = utils.default_recording_name(rec_dir)
    system_info = android_system_info(info_json)

    # Create a recording info file with the new format,
    # fill out the information, validate, and return.
    new_info_file = RecordingInfoFile.create_empty_file(rec_dir, Version("2.1"))
    new_info_file.recording_uuid = recording_uuid
    new_info_file.start_time_system_ns = start_time_system_ns
    new_info_file.start_time_synced_ns = start_time_synced_ns
    new_info_file.duration_ns = duration_ns
    new_info_file.recording_software_name = recording_software_name
    new_info_file.recording_software_version = recording_software_version
    new_info_file.recording_name = recording_name
    new_info_file.system_info = system_info
    new_info_file.validate()
    new_info_file.save_file()


def _rename_pi_files(recording: PupilRecording):
    for path in recording.files():
        # replace prefix based on cam_type, need to reformat part number
        match = re.match(
            r"^(?P<prefix>PI (?P<cam_type>left|right|world) v\d+ ps(?P<part>\d+))",
            path.name,
        )
        if match:
            replacement_for_cam_type = {
                "right": "eye0",
                "left": "eye1",
                "world": "world",
            }
            replacement = replacement_for_cam_type[match.group("cam_type")]
            part_number = int(match.group("part"))
            if part_number > 1:
                # add zero-filled part number - 1
                # NOTE: recordings for PI start at part 1, mobile start at part 0
                replacement += f"_{part_number - 1:03}"

            new_name = path.name.replace(match.group("prefix"), replacement)
            path.replace(path.with_name(new_name))  # rename with overwrite


def _rewrite_timestamps(recording: PupilRecording):
    start_time = recording.meta_info.start_time_synced_ns

    def conversion(timestamps: np.array):
        # Subtract start_time from all times in the recording, so timestamps
        # start at 0. This is to increase precision when converting
        # timestamps to float32, e.g. for OpenGL!
        SECONDS_PER_NANOSECOND = 1e-9
        return (timestamps - start_time) * SECONDS_PER_NANOSECOND

    update_utils._rewrite_times(recording, dtype="<u8", conversion=conversion)


def _convert_gaze(recording: PupilRecording):
    width, height = 1088, 1080

    logger.info("Converting gaze data...")
    template_datum = {
        "topic": "gaze.pi",
        "norm_pos": None,
        "timestamp": None,
        "confidence": 1.0,
    }
    with fm.PLData_Writer(recording.rec_dir, "gaze") as writer:
        for ((x, y), ts) in pi_gaze_items(root_dir=recording.rec_dir):
            template_datum["timestamp"] = ts
            template_datum["norm_pos"] = m.normalize(
                (x, y), size=(width, height), flip_y=True
            )
            writer.append(template_datum)
        logger.info(f"Converted {len(writer.ts_queue)} gaze positions.")


def android_system_info(info_json: dict) -> str:
    android_device_id = info_json.get("android_device_id", "?")
    android_device_name = info_json.get("android_device_name", "?")
    android_device_model = info_json.get("android_device_model", "?")
    return (
        f"Android device ID: {android_device_id}; "
        f"Android device name: {android_device_name}; "
        f"Android device model: {android_device_model}"
    )
