"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from .. import Version
from .recording_info import RecordingInfoFile
from .recording_info_2_0 import _RecordingInfoFile_2_0

RecordingInfoFile.register_child_class(Version("2.0"), _RecordingInfoFile_2_0)
