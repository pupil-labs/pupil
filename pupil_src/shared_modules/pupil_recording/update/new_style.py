import logging
from pathlib import Path

import file_methods as fm
from version_utils import get_version

from .. import Version
from ..info import RecordingInfoFile
from ..recording import PupilRecording
from ..recording_utils import InvalidRecordingException

logger = logging.getLogger(__name__)


def recording_update_to_latest_new_style(rec_dir: str):
    info_file = RecordingInfoFile.read_file_from_recording(rec_dir)

    # TODO: get_version() returns a LooseVersion, but we are using packaging.Version
    # now, need to adjust this across the codebase
    if info_file.min_player_version > Version(get_version().vstring):
        player_out_of_date = (
            "Recording requires a newer version of Player: "
            f"{info_file.min_player_version}"
        )
        raise InvalidRecordingException(reason=player_out_of_date)

    # if info_file.min_player_version < Version("1.17"):
    #     ... incremental update ...
    #     Increases min_player_version


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
