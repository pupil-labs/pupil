"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import namedtuple

from storage import StorageItem

# this plugin does not care about the content of the result, it just receives it from
# the calibration routine and handles it to the gaze mapper
CalibrationResult = namedtuple(
    "CalibrationResult", ["mapping_plugin_name", "mapper_args"]
)


class Calibration(StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        recording_uuid,
        mapping_method,
        frame_index_range,
        minimum_confidence,
        status="Not calculated yet",
        is_offline_calibration=True,
        result=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.recording_uuid = recording_uuid
        self.mapping_method = mapping_method
        self.frame_index_range = frame_index_range
        self.minimum_confidence = minimum_confidence
        self.status = status
        self.is_offline_calibration = is_offline_calibration
        if result is None or isinstance(result, CalibrationResult):
            self.result = result
        else:
            # when reading from files, we receive a list with the result data.
            # This logic actually belongs to 'from_tuple', but it's here because
            # otherwise 'from_tuple' would become much uglier
            self.result = CalibrationResult(*result)

    @staticmethod
    def from_tuple(tuple_):
        return Calibration(*tuple_)

    @property
    def as_tuple(self):
        return (
            self.unique_id,
            self.name,
            self.recording_uuid,
            self.mapping_method,
            self.frame_index_range,
            self.minimum_confidence,
            self.status,
            self.is_offline_calibration,
            self.result,
        )
