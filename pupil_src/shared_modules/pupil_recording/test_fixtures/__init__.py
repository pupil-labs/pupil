"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import enum
import json
import os
import tempfile
import typing as T
import uuid

import csv_utils


def info_csv_v1_14() -> str:
    return _create_dir_with_info_file(
        info_type=_InfoFileType.CSV,
        info_dict={
            "Recording Name": "2019_04_01",
            "Start Date": "01.04.2019",
            "Start Time": "16:08:20",
            "Start Time (System)": "1554127700.34813",
            "Start Time (Synced)": "23849.548983486",
            "Recording UUID": "3361ce29-b43d-44fc-9f2f-1861ae0f2a5b",
            "Duration Time": "00:00:59",
            "World Camera Frames": "1751",
            "World Camera Resolution": "1280x720",
            "Capture Software Version": "1.11.10",
            "Data Format Version": "v1.14",
            "System Info": "User: roman, Platform: Linux, Machine: romans-nuc, Release: 4.18.0-16-generic, Version: #17~18.04.1-Ubuntu SMP Tue Feb 12 13:35:51 UTC 2019",
        },
    )


def info_json_basic() -> str:
    return _create_dir_with_info_file(
        info_type=_InfoFileType.JSON,
        info_dict={
    	    "recording_uuid": "8922eb87-46dc-4fcb-b6a1-bf9aedf442fe",
	        "app_version": "0.5.5",
	        "data_format_version": "1.0",
	        "android_device_model": "OnePlus 6",
    	    "android_device_name": "C",
	        "duration": 14442000000,
	        "glasses_serial_number": "7l26t",
	        "scene_camera_serial_number": "j9782",
	        "start_time": 1566406635733000000,
	        "start_time_synced": 1566406635733000000,
	        "template_data": {
		        "notes": "The burger was unsatisfactory",
		        "recording_name": "Recording",
		        "id": "e36ee102-569c-11e9-8647-d663bd873d93"
	        },
    	    "wearer_id": "50577997-f5b6-4d44-af6e-c53c2f5c32a4"
        },
    )


########## PRIVATE ##########


@enum.unique
class _InfoFileType(enum.Enum):
    CSV = "csv"
    JSON = "json"

    @property
    def file_name(self) -> str:
        if self == _InfoFileType.CSV:
            return "info.csv"
        if self == _InfoFileType.JSON:
            return "info.json"
        else:
            raise ValueError(f"Unknown info file type: {self}")

    def write_info_to_file(self, info_dict: dict, file):
        if self == _InfoFileType.CSV:
            csv_utils.write_key_value_file(file, info_dict, append=False)
        elif self == _InfoFileType.JSON:
            json.dump(info_dict, file)
        else:
            raise ValueError(f"Unknown info file type: {self}")


def _create_dir_with_info_file(info_type: _InfoFileType, info_dict: dict) -> str:

    root_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    file_path = os.path.join(root_dir, info_type.file_name)

    os.makedirs(root_dir)

    assert not os.path.exists(file_path)
    with open(file_path, "w") as file:
        info_type.write_info_to_file(info_dict=info_dict, file=file)
    assert os.path.isfile(file_path)

    return root_dir
