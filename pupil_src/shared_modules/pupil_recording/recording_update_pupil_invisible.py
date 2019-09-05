from .info import recording_info_utils as utils
from .info import RecordingInfoFile


def is_pupil_invisible_recording(rec_dir: str) -> bool:
    info_csv = utils.read_info_csv_file(rec_dir)
    try:
        return info_csv["Capture Software"] == "Pupil Invisible" and "Data Format Version" not in info_csv
    except KeyError:
        return False


def recording_update_pupil_invisible_to_pprf_2_0(rec_dir: str) -> RecordingInfoFile:
    _recording_update_pupil_invisible_to_v1_15(rec_dir)
    return _recording_update_pupil_invisible_from_v1_15_to_pprf_2_0(rec_dir)


def _recording_update_pupil_invisible_to_v1_15(rec_dir: str):
    pass #TODO: Update Pupil Invisible recording to Pupil Capture v1.15 format


def _recording_update_pupil_invisible_from_v1_15_to_pprf_2_0(rec_dir: str) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)
    info_json = utils.read_info_json_file(rec_dir)

    # Get information about recording from info.csv and info.json
    recording_uuid = info_json["recording_uuid"]
    start_time_system_ns = int(info_json["start_time"])
    start_time_synced_ns = int(info_json["start_time_synced"])
    duration_ns = int(info_json["duration"])
    recording_software_name = RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
    recording_software_version = utils.recording_version_from_string(info_json["app_version"])
    recording_name = utils.default_recording_name(rec_dir)
    system_info = android_system_info(info_json)

    # Create a recording info file with the new format, fill out the information, validate, and return.
    new_info_file = RecordingInfoFile.create_empty_file(rec_dir)
    new_info_file.recording_uuid = recording_uuid
    new_info_file.start_time_system_ns = start_time_system_ns
    new_info_file.start_time_synced_ns = start_time_synced_ns
    new_info_file.duration_ns = duration_ns
    new_info_file.recording_software_name = recording_software_name
    new_info_file.recording_software_version = recording_software_version
    new_info_file.recording_name = recording_name
    new_info_file.system_info = system_info
    new_info_file.validate()
    return new_info_file


def android_system_info(info_json: dict) -> str:
    android_device_id = info_json.get("android_device_id", "?")
    android_device_name = info_json.get("android_device_name", "?")
    android_device_model = info_json.get("android_device_model", "?")
    return f"Android device ID: {android_device_id}; Android device name: {android_device_name}; Android device model: {android_device_model}"
