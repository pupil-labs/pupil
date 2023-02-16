"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from version_utils import ParsedVersion, parse_version

from . import RecordingInfoFile
from . import recording_info_utils as utils
from .recording_info_2_0 import _RecordingInfoFile_2_0


class _RecordingInfoFile_2_2(_RecordingInfoFile_2_0):
    # Used to make Pupil v2.0 recordings backwards incompatible with v1.*

    @property
    def meta_version(self) -> ParsedVersion:
        return parse_version("2.2")

    @property
    def min_player_version(self) -> ParsedVersion:
        return parse_version("2.0")

    @property
    def _private_key_schema(self) -> RecordingInfoFile._KeyValueSchema:
        return {
            **super()._private_key_schema,
            # overwrite meta_version key from parent
            "meta_version": (utils.validator_version_string, lambda _: "2.2"),
            "min_player_version": (utils.validator_version_string, lambda _: "2.0"),
        }
