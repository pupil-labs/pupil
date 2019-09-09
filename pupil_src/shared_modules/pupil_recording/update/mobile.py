import uuid

from pupil_recording.info import recording_info_utils as utils
from pupil_recording.info.recording_info import RecordingInfoFile, Version
from pupil_recording.recording_utils import InvalidRecordingException

NEXT_UNSUPPORTED_VERSION = Version("1.3")


def transform_mobile_to_corresponding_new_style(rec_dir: str) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)
    mobile_version = Version(info_csv["Data Format Version"])

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
    # TODO: rename info.csv file to info.mobile.csv
    # TODO: rename and convert time, video


def _generate_pprf_2_0_info_file(rec_dir: str) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)

    # Get information about recording from info.csv
    recording_uuid = info_csv.get("Recording UUID", uuid.uuid4())
    start_time_system_s = float(info_csv["Start Time (System)"])
    start_time_synced_s = float(info_csv["Start Time (Synced)"])
    duration_s = float(info_csv["Duration Time"])
    recording_software_name = info_csv["Capture Software"]
    recording_software_version = Version(info_csv["Capture Software Version"])
    recording_name = info_csv.get(
        "Recording Name", utils.default_recording_name(rec_dir)
    )
    system_info = info_csv.get("System Info", utils.default_system_info(rec_dir))

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
