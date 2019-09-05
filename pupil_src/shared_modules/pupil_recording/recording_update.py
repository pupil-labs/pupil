import os
from .info import RecordingInfoFile
from .recording_update_legacy import (
    was_recording_opened_in_player_before,
    recording_update_legacy_to_pprf_2_0,
)
from .recording_update_pupil_invisible import (
    is_pupil_invisible_recording,
    recording_update_pupil_invisible_to_pprf_2_0,
)
from .recording_update_pupil_mobile import (
    is_pupil_mobile_recording,
    recording_update_pupil_mobile_to_pprf_2_0,
)


def recording_update(rec_dir: str) -> RecordingInfoFile:

    if RecordingInfoFile.does_recording_contain_info_file(rec_dir):
        info_file = RecordingInfoFile.read_file_from_recording(rec_dir)

    elif was_recording_opened_in_player_before(rec_dir):
        info_file = recording_update_legacy_to_pprf_2_0(rec_dir)

    elif is_pupil_invisible_recording(rec_dir):
        info_file = recording_update_pupil_invisible_to_pprf_2_0(rec_dir)

    elif is_pupil_mobile_recording(rec_dir):
        info_file = recording_update_pupil_mobile_to_pprf_2_0(rec_dir)

    else:
        info_file = recording_update_legacy_to_pprf_2_0(rec_dir)

    info_file = info_file.updated_file()
    info_file.save_file()
    return info_file
