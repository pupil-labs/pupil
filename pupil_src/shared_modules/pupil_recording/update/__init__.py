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

    if recording_type == RecordingType.INVISIBLE:
        # NOTE: there is an issue with PI recordings, where sometimes multiple parts of
        # the recording are stored as an .mjpeg and .mp4, but for the same part number.
        # The recording is un-usable in this case, since the time information is lost.
        # Trying to open the recording will crash in the lookup-table generation. We
        # just gracefully exit here and display an error message.
        PI_world_videos = PupilRecording.FileFilter(rec_dir).pi().world().videos()
        name_stems = [path.stem for path in PI_world_videos]
        unique_name_stems = set(name_stems)
        # the assumption here is that we should have only one file per name stem
        if len(name_stems) != len(unique_name_stems):
            duplicate_stems = [
                unique_stem
                for unique_stem in unique_name_stems
                if name_stems.count(unique_stem) > 1
            ]
            duplicate_videos = [
                path.name for path in PI_world_videos if path.stem in duplicate_stems
            ]
            logger.error(
                "Found duplicate video stems for this Pupil Invisible recording!"
                " Please each out to info@pupil-labs.com for support!\n"
                f"Duplicate videos: {', '.join(duplicate_videos)}"
            )
            raise InvalidRecordingException(
                "This recording cannot be opened in Player.",
                "Please each out to info@pupil-labs.com for support!",
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
        File_Source(
            SimpleNamespace(), source_path=videos[0], fill_gaps=True, timing=None
        )


__all__ = ["update_recording"]
