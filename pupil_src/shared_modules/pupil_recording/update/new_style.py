"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
from pathlib import Path

import file_methods as fm
from version_utils import get_version

from .. import Version
from ..info import RecordingInfoFile
from ..recording import PupilRecording
from ..recording_utils import InvalidRecordingException
from . import invisible

logger = logging.getLogger(__name__)


def recording_update_to_latest_new_style(rec_dir: str):
    info_file = RecordingInfoFile.read_file_from_recording(rec_dir)
    check_min_player_version(info_file)

    # incremental upgrade ...
    if info_file.meta_version < Version("2.1"):
        info_file = update_newstyle_20_21(rec_dir)
    if info_file.meta_version < Version("2.2"):
        info_file = update_newstyle_21_22(rec_dir)


def check_min_player_version(info_file: RecordingInfoFile):
    # TODO: get_version() returns a LooseVersion, but we are using packaging.Version
    # now, need to adjust this across the codebase
    if info_file.min_player_version > Version(get_version().vstring):
        player_out_of_date = (
            "Recording requires a newer version of Player: "
            f"{info_file.min_player_version}"
        )
        raise InvalidRecordingException(reason=player_out_of_date)


def check_for_worldless_recording_new_style(rec_dir):
    logger.info("Checking for world-less recording...")
    rec_dir = Path(rec_dir)

    recording = PupilRecording(rec_dir)
    world_videos = recording.files().core().world().videos()
    if not world_videos:
        logger.info("No world video found. Constructing an artificial replacement.")
        fake_world_version = 2
        fake_world_object = {"version": fake_world_version}
        fake_world_path = rec_dir / "world.fake"
        fm.save_object(fake_world_object, fake_world_path)


def update_newstyle_20_21(rec_dir: str):
    # There was a bug in v1.16 and v1.17 that caused corrupted gaze data when
    # transforming a PI recording to new_style. Need to delete and re-transform.
    info_file = RecordingInfoFile.read_file_from_recording(rec_dir)

    if (
        info_file.recording_software_name
        == RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
    ):
        logger.debug("Upgrading PI recording opened with pre v.1.18")
        for path in Path(rec_dir).iterdir():
            if not path.is_file:
                continue
            if path.name in ["gaze.pldata", "gaze_timestamps.npy"]:
                logger.debug(
                    f"Deleting potentially corrupted file '{path.name}'"
                    f" from pre v1.18."
                )
                path.unlink()
        invisible._convert_gaze(PupilRecording(rec_dir))
        # Bump info file version to 2.1
        new_info_file = RecordingInfoFile.create_empty_file(
            rec_dir, fixed_version=Version("2.1")
        )
        new_info_file.update_writeable_properties_from(info_file)
        info_file = new_info_file
        info_file.save_file()

    return info_file


def update_newstyle_21_22(rec_dir: str):
    # Used to make Pupil v2.0 recordings backwards incompatible with v1.x
    old_info_file = RecordingInfoFile.read_file_from_recording(rec_dir)
    new_info_file = RecordingInfoFile.create_empty_file(
        rec_dir, fixed_version=Version("2.2")
    )
    new_info_file.update_writeable_properties_from(old_info_file)
    new_info_file.save_file()
    return new_info_file
