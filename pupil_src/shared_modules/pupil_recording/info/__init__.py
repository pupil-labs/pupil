"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from version_utils import parse_version

from .recording_info import RecordingInfo, RecordingInfoFile
from .recording_info_2_0 import _RecordingInfoFile_2_0
from .recording_info_2_1 import _RecordingInfoFile_2_1
from .recording_info_2_2 import _RecordingInfoFile_2_2
from .recording_info_2_3 import _RecordingInfoFile_2_3

RecordingInfoFile.register_child_class(parse_version("2.0"), _RecordingInfoFile_2_0)
RecordingInfoFile.register_child_class(parse_version("2.1"), _RecordingInfoFile_2_1)
RecordingInfoFile.register_child_class(parse_version("2.2"), _RecordingInfoFile_2_2)
RecordingInfoFile.register_child_class(parse_version("2.3"), _RecordingInfoFile_2_3)
