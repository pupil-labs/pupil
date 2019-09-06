import os
from .info import RecordingInfoFile
from .old_style import (
    was_recording_opened_in_player_before,
    recording_update_old_style_to_pprf_2_0,
)
from .pupil_invisible import (
    is_pupil_invisible_recording,
    transform_PI_to_corresponding_new_style,
)
from .pupil_mobile import (
    is_pupil_mobile_recording,
    transform_mobile_to_corresponding_new_style,
)
from .new_style import recording_update_to_latest_new_style

from ..recording import get_recording_type, RecordingType

_transformations_to_new_style = {
    RecordingType.INVISIBLE: transform_PI_to_corresponding_new_style,
    RecordingType.MOBILE: transform_mobile_to_corresponding_new_style,
    RecordingType.OLD_STYLE: recording_update_old_style_to_pprf_2_0,
}


def update_recording(rec_dir: str):

    recording_type = get_recording_type(rec_dir)
    if recording_type in _transformations_to_new_style:
        _transformations_to_new_style[recording_type](rec_dir)

    # update to latest
    recording_update_to_latest_new_style(rec_dir)
