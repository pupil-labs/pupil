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

import tasklib
from gaze_producer import worker
from observable import Observable

logger = logging.getLogger(__name__)


class CalibrationController(Observable):
    def __init__(
        self,
        calibration_storage,
        reference_location_storage,
        task_manager,
        get_current_trim_mark_range,
        recording_uuid,
    ):
        self._calibration_storage = calibration_storage
        self._reference_location_storage = reference_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._recording_uuid = str(recording_uuid)

    def calculate(self, calibration):
        def on_calibration_completed(status_and_result):
            calibration.status, result = status_and_result
            if result is not None:
                calibration.gazer_class_name = result.gazer_class_name
                calibration.update(calib_params=result.params)
                self._calibration_storage.save_to_disk()
                self.on_calibration_computed(calibration)

        calibration.status = "Recalculating..."
        if len(self._reference_location_storage.items) == 0:
            error_message = f"You first need to detect reference locations before calculating the calibration '{calibration.name}'"
            self._abort_calculation(calibration, error_message)
            return None
        task = worker.create_calibration.create_task(
            calibration, all_reference_locations=self._reference_location_storage
        )
        task.add_observer("on_completed", on_calibration_completed)
        task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(
            task, identifier=f"{calibration.unique_id}-calibration"
        )
        return task

    def on_calibration_computed(self, calibration):
        pass

    def on_calculation_could_not_be_started(self):
        pass

    def set_calibration_range_from_current_trim_marks(self, calibration):
        calibration.frame_index_range = self._get_current_trim_mark_range()

    def is_from_same_recording(self, calibration):
        """
        False if the calibration file was copied from another recording directory
        """
        return (
            calibration is not None
            and calibration.recording_uuid == self._recording_uuid
        )

    def _abort_calculation(self, calibration, error_message):
        logger.error(error_message)
        calibration.status = error_message
        self.on_calculation_could_not_be_started()
