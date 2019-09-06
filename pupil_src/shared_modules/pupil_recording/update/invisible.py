from ..info import recording_info_utils as utils
from ..info import RecordingInfoFile, Version
from ..recording import InvalidRecordingException


NEWEST_SUPPORTED_VERSION = Version("1.0")


def is_pupil_invisible_recording(rec_dir: str) -> bool:
    try:
        utils.read_info_json_file(rec_dir)
        return True
    except FileNotFoundError:
        return False


def transform_invisible_to_corresponding_new_style(rec_dir: str):
    info_json = utils.read_info_json_file(rec_dir)
    pi_version = Version(info_json["data_format_version"])

    if pi_version > NEWEST_SUPPORTED_VERSION:
        raise InvalidRecordingException(
            f"This version of Pupil Invisible is too new : {pi_version}"
        )

    # elif pi_version > 3.0:
    #     ...
    # elif pi_version > 2.0:
    #     ...

    else:
        _transform_invisible_v1_0_to_pprf_2_0(rec_dir)


def _transform_invisible_v1_0_to_pprf_2_0(rec_dir: str):
    _generate_pprf_2_0_info_file(rec_dir)
    # TODO: rename info.json file to info.invisible.json
    # TODO: rename and convert time, video, gaze


def _generate_pprf_2_0_info_file(rec_dir: str) -> RecordingInfoFile:
    info_json = utils.read_info_json_file(rec_dir)

    # Get information about recording from info.csv and info.json
    recording_uuid = info_json["recording_uuid"]
    start_time_system_ns = int(info_json["start_time"])
    start_time_synced_ns = int(info_json["start_time_synced"])
    duration_ns = int(info_json["duration"])
    recording_software_name = RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
    recording_software_version = Version(info_json["app_version"])
    recording_name = utils.default_recording_name(rec_dir)
    system_info = android_system_info(info_json)

    # Create a recording info file with the new format,
    # fill out the information, validate, and return.
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
    new_info_file.save_file()


def android_system_info(info_json: dict) -> str:
    android_device_id = info_json.get("android_device_id", "?")
    android_device_name = info_json.get("android_device_name", "?")
    android_device_model = info_json.get("android_device_model", "?")
    return (
        f"Android device ID: {android_device_id}; "
        f"Android device name: {android_device_name}; "
        f"Android device model: {android_device_model}"
    )
