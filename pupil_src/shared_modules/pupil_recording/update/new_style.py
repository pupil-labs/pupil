from pupil_recording.info.recording_info import RecordingInfoFile, Version
from pupil_recording.recording_utils import InvalidRecordingException
from version_utils import pupil_version


def recording_update_to_latest_new_style(rec_dir: str):
    info_file = RecordingInfoFile.read_file_from_recording(rec_dir)

    if info_file.min_player_version > Version(pupil_version()):
        player_out_of_date = (
            "Recording requires a newer version of Player: "
            f"{info_file.min_player_version}"
        )
        raise InvalidRecordingException(reason=player_out_of_date)

    # if info_file.min_player_version < Version("1.17"):
    #     ... incremental update ...
    #     Increases min_player_version
