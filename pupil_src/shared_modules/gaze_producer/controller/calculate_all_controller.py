"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class CalculateAllController:
    def __init__(
        self,
        reference_detection_controller,
        reference_location_storage,
        calibration_controller,
        calibration_storage,
        gaze_mapper_controller,
        gaze_mapper_storage,
    ):
        self._reference_detection_controller = reference_detection_controller
        self._reference_location_storage = reference_location_storage
        self._calibration_controller = calibration_controller
        self._calibration_storage = calibration_storage
        self._gaze_mapper_controller = gaze_mapper_controller
        self._gaze_mapper_storage = gaze_mapper_storage

    def calculate_all(self):
        """
        (Re)Calculate all calibrations and gaze mappings with their respective
        current settings. If there are no reference locations in the storage,
        first the current reference detector is run.
        """
        if self.does_detect_references:
            task = self._reference_detection_controller.start_detection()
            task.add_observer("on_completed", self._on_reference_detection_completed)
        else:
            self._calculate_all_calibrations()

    @property
    def does_detect_references(self):
        """
        True if the controller would first detect reference locations in calculate_all()
        """
        at_least_one_reference_location = any(
            True for _ in self._reference_location_storage
        )
        return not at_least_one_reference_location

    def _on_reference_detection_completed(self, _):
        self._calculate_all_calibrations()

    def _calculate_all_calibrations(self):
        for calibration in self._calibration_storage:
            calculation_possible = (
                self._calibration_controller.is_from_same_recording(calibration)
                and calibration.is_offline_calibration
            )
            if calculation_possible:
                task = self._calibration_controller.calculate(calibration)
                task.add_observer(
                    "on_completed",
                    self._create_calibration_complete_handler(calibration),
                )
            else:
                self._calculate_gaze_mappers_based_on_calibration(calibration)

    def _create_calibration_complete_handler(self, calibration):
        return lambda _: self._calculate_gaze_mappers_based_on_calibration(calibration)

    def _calculate_gaze_mappers_based_on_calibration(self, calibration):
        for gaze_mapper in self._gaze_mapper_storage:
            if gaze_mapper.calibration_unique_id == calibration.unique_id:
                self._gaze_mapper_controller.calculate(gaze_mapper)
