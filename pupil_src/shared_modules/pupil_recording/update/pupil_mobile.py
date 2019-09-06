import uuid

from .info import recording_info_utils as utils
from .info import RecordingInfoFile


def is_pupil_mobile_recording(rec_dir: str) -> bool:
    info_csv = utils.read_info_csv_file(rec_dir)
    try:
        return (
            info_csv["Capture Software"] == "Pupil Mobile"
            and "Data Format Version" not in info_csv
        )
    except KeyError:
        return False


def transform_mobile_to_corresponding_new_style(rec_dir: str) -> RecordingInfoFile:
    _recording_update_pupil_mobile_to_v1_15(rec_dir)
    return _recording_update_pupil_mobile_from_v1_15_to_pprf_2_0(rec_dir)


def _recording_update_pupil_mobile_to_v1_15(rec_dir: str):
    pass  # TODO: Update Pupil Invisible recording to Pupil Capture v1.15 format


def _recording_update_pupil_mobile_from_v1_15_to_pprf_2_0(
    rec_dir: str
) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)

    # Get information about recording from info.csv
    recording_uuid = info_csv.get("Recording UUID", uuid.uuid4())
    start_time_system_s = float(info_csv["Start Time (System)"])
    start_time_synced_s = float(info_csv["Start Time (Synced)"])
    duration_s = float(info_csv["Duration Time"])
    recording_software_name = info_csv["Capture Software"]
    recording_software_version = utils.recording_version_from_string(
        info_csv["Capture Software Version"]
    )
    recording_name = info_csv.get(
        "Recording Name", utils.default_recording_name(rec_dir)
    )
    system_info = info_csv.get("System Info", utils.default_system_info(rec_dir))

    # Create a recording info file with the new format, fill out the information, validate, and return.
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
    return new_info_file
