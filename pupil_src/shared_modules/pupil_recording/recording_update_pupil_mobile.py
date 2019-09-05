from .info import recording_info_utils as utils
from .info import RecordingInfoFile


def is_pupil_mobile_recording(rec_dir: str) -> bool:
    info_csv = utils.read_info_csv_file(rec_dir)
    try:
        return info_csv["Capture Software"] == "Pupil Mobile" and "Data Format Version" not in info_csv
    except KeyError:
        return False


def recording_update_pupil_mobile_to_pprf_2_0(rec_dir: str) -> RecordingInfoFile:
    info_csv = utils.read_info_csv_file(rec_dir)

    # Get information about recording from info.csv
    recording_uuid = None #TODO
    start_time_system_s = None  # TODO
    start_time_synced_s = None  # TODO
    duration_s = None  # TODO
    recording_software_name = None  # TODO
    recording_software_version = None  # TODO
    recording_name = None  # TODO
    system_info = None #TODO

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
