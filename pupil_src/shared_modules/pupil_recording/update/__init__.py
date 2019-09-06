from pupil_recording.recording import RecordingType, get_recording_type
from pupil_recording.update.invisible import (
    transform_invisible_to_corresponding_new_style,
)
from pupil_recording.update.mobile import transform_mobile_to_corresponding_new_style
from pupil_recording.update.new_style import recording_update_to_latest_new_style
from pupil_recording.update.old_style import recording_update_old_style_to_pprf_2_0

_transformations_to_new_style = {
    RecordingType.INVISIBLE: transform_invisible_to_corresponding_new_style,
    RecordingType.MOBILE: transform_mobile_to_corresponding_new_style,
    RecordingType.OLD_STYLE: recording_update_old_style_to_pprf_2_0,
}


def update_recording(rec_dir: str):

    recording_type = get_recording_type(rec_dir)
    if recording_type in _transformations_to_new_style:
        _transformations_to_new_style[recording_type](rec_dir)

    # TODO: Check worldless recording

    # update to latest
    recording_update_to_latest_new_style(rec_dir)


__all__ = ["update_recording"]
