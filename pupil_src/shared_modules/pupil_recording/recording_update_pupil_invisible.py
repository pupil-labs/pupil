from .info import recording_info_utils as utils
from .info import RecordingInfoFile


def is_pupil_invisible_recording(rec_dir: str) -> bool:
    info_csv = utils.read_info_csv_file(rec_dir)
    try:
        return info_csv["Capture Software"] == "Pupil Invisible" and "Data Format Version" not in info_csv
    except KeyError:
        return False


def recording_update_pupil_invisible_to_pprf_2_0(rec_dir: str) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)
    info_json = utils.read_info_json_file(rec_dir)

    # Get information about recording from info.csv and info.json
    recording_uuid = None #TODO
    start_time_system_ns = None  # TODO
    start_time_synced_ns = None  # TODO
    duration_ns = None  # TODO
    recording_software_name = None  # TODO
    recording_software_version = None  # TODO
    recording_name = None  # TODO
    system_info = None #TODO

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
