"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from storage import StorageItem


class GazeMapper(StorageItem):
    version = 1

    def __init__(
        self,
        unique_id,
        name,
        calibration_unique_id,
        mapping_index_range,
        validation_index_range,
        validation_outlier_threshold_deg,
        manual_correction_x=0.0,
        manual_correction_y=0.0,
        activate_gaze=True,
        status="Not calculated yet",
        accuracy_result="",
        precision_result="",
        gaze=[],
        gaze_ts=[],
    ):
        self.unique_id = unique_id
        self.name = name
        self.calibration_unique_id = calibration_unique_id
        self.mapping_index_range = tuple(mapping_index_range)
        self.validation_index_range = tuple(validation_index_range)
        self.validation_outlier_threshold_deg = validation_outlier_threshold_deg
        self.manual_correction_x = manual_correction_x
        self.manual_correction_y = manual_correction_y
        self.activate_gaze = activate_gaze
        self.status = status
        self.accuracy_result = accuracy_result
        self.precision_result = precision_result
        self.gaze = gaze
        self.gaze_ts = gaze_ts

    @property
    def calculate_complete(self):
        # we cannot just use `self.gaze and self.gaze_ts` because this ands the arrays
        return len(self.gaze) > 0 and len(self.gaze_ts) > 0

    @staticmethod
    def from_tuple(tuple_):
        return GazeMapper(*tuple_)

    @property
    def as_tuple(self):
        return (
            self.unique_id,
            self.name,
            self.calibration_unique_id,
            self.mapping_index_range,
            self.validation_index_range,
            self.validation_outlier_threshold_deg,
            self.manual_correction_x,
            self.manual_correction_y,
            self.activate_gaze,
            self.status,
            self.accuracy_result,
            self.precision_result,
        )
