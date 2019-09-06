import os
import uuid

from .recording_info import (
    Version,
    RecordingInfo,
    RecordingInfoFile,
    RecordingInfoInvalidError,
)

from . import test_fixtures


def test_info_json_basic():
    rec_dir = test_fixtures.info_json_basic()
    info_file = RecordingInfoFile.read_json(rec_dir=rec_dir)
    _check_info_property_types(info_file)

    assert len(info_file.recording_name) > 0
    assert info_file.software_version.release == (0, 5, 5)
    assert info_file.data_format_version.release == (1, 0)


def test_info_csv_v1_14():
    rec_dir = test_fixtures.info_csv_v1_14()
    info_file = RecordingInfoFile.read_csv(rec_dir=rec_dir)
    _check_info_property_types(info_file)

    assert len(info_file.recording_name) > 0
    assert info_file.software_version.release == (1, 11, 10)
    assert info_file.data_format_version.release == (1, 14)


########## PRIVATE ##########


def _check_info_property_types(info: RecordingInfo):
    assert isinstance(info.recording_uuid, uuid.UUID)
    assert isinstance(info.recording_name, str)
    assert isinstance(info.software_version, Version)
    assert isinstance(info.data_format_version, Version)
    assert isinstance(info.duration_s, float)
    assert isinstance(info.duration_ns, int)
    assert isinstance(info.start_time_s, float)
    assert isinstance(info.start_time_ns, int)
    assert isinstance(info.start_time_synced_s, float)
    assert isinstance(info.start_time_synced_ns, int)
    assert isinstance(info.world_camera_frames, int)
    assert isinstance(info.world_camera_resolution, tuple)
