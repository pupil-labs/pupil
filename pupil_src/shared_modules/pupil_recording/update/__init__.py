"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
from types import SimpleNamespace

from video_capture.file_backend import File_Source

from ..recording import PupilRecording
from ..recording_utils import (
    InvalidRecordingException,
    RecordingType,
    get_recording_type,
)
from .invisible import transform_invisible_to_corresponding_new_style
from .mobile import transform_mobile_to_corresponding_new_style
from .new_style import (
    check_for_worldless_recording_new_style,
    recording_update_to_latest_new_style,
)
from .old_style import transform_old_style_to_pprf_2_0

logger = logging.getLogger(__name__)

_transformations_to_new_style = {
    RecordingType.INVISIBLE: transform_invisible_to_corresponding_new_style,
    RecordingType.MOBILE: transform_mobile_to_corresponding_new_style,
    RecordingType.OLD_STYLE: transform_old_style_to_pprf_2_0,
}


def update_recording(rec_dir: str):

    recording_type = get_recording_type(rec_dir)

    if recording_type == RecordingType.CLOUD_CSV_EXPORT:
        raise InvalidRecordingException(
            "Pupil Player does not support\nPupil Cloud CSV exports"
        )
    if recording_type == RecordingType.NEON:
        raise InvalidRecordingException(
            "Pupil Player does not support\nNeon Companion recordings"
        )

    if recording_type == RecordingType.INVISIBLE:
        # NOTE: there is an issue with PI recordings, where sometimes multiple parts of
        # the recording are stored as an .mjpeg and .mp4, but for the same part number.
        # The recording is un-usable in this case, since the time information is lost.
        # Trying to open the recording will crash in the lookup-table generation. We
        # just gracefully exit here and display an error message.
        mjpeg_world_videos = (
            PupilRecording.FileFilter(rec_dir).pi().world().filter_patterns(".mjpeg$")
        )
        if mjpeg_world_videos:
            videos = [
                path.name
                for path in PupilRecording.FileFilter(rec_dir).pi().world().videos()
            ]
            logger.error(
                "Found mjpeg world videos for this Pupil Invisible recording! Videos:\n"
                + ",\n".join(videos)
            )
            raise InvalidRecordingException(
                "This recording cannot be opened in Player.",
                recovery="Please reach out to info@pupil-labs.com for support!",
            )

    if recording_type in _transformations_to_new_style:
        _transformations_to_new_style[recording_type](rec_dir)

    _assert_compatible_meta_version(rec_dir)

    check_for_worldless_recording_new_style(rec_dir)

    # update to latest
    recording_update_to_latest_new_style(rec_dir)

    # generate lookup tables once at the start of player, so we don't pause later for
    # compiling large lookup tables when they are needed
    _generate_all_lookup_tables(rec_dir)

    # Update offline calibrations to latest calibration model version
    from gaze_producer.model.legacy import update_offline_calibrations_to_latest_version

    update_offline_calibrations_to_latest_version(rec_dir)


def _assert_compatible_meta_version(rec_dir: str):
    # This will throw InvalidRecordingException if we cannot open the recording due
    # to meta info version or min_player_version mismatches.
    PupilRecording(rec_dir)


def _generate_all_lookup_tables(rec_dir: str):
    recording = PupilRecording(rec_dir)
    videosets = [
        recording.files().core().world().videos(),
        recording.files().core().eye0().videos(),
        recording.files().core().eye1().videos(),
    ]
    for videos in videosets:
        if not videos:
            continue
        source_path = videos[0].resolve()
        File_Source(
            SimpleNamespace(), source_path=source_path, fill_gaps=True, timing=None
        )


__all__ = ["update_recording"]
