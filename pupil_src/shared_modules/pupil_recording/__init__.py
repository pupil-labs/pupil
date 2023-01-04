"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .info import RecordingInfo, RecordingInfoFile
from .recording import PupilRecording
from .recording_utils import InvalidRecordingException, assert_valid_recording_type
