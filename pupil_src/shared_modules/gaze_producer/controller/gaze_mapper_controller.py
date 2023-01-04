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
from itertools import chain

import player_methods
import tasklib
from gaze_producer import worker
from observable import Observable

logger = logging.getLogger(__name__)


class GazeMapperController(Observable):
    def __init__(
        self,
        gaze_mapper_storage,
        calibration_storage,
        reference_location_storage,
        task_manager,
        get_current_trim_mark_range,
        publish_gaze_bisector,
    ):
        self._gaze_mapper_storage = gaze_mapper_storage
        self._calibration_storage = calibration_storage
        self._reference_location_storage = reference_location_storage
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._publish_gaze_bisector = publish_gaze_bisector

        self._gaze_mapper_storage.add_observer("delete", self.on_gaze_mapper_deleted)

        # make mappings loaded from disk known to Player
        self.publish_all_enabled_mappers()

    def set_mapping_range_from_current_trim_marks(self, gaze_mapper):
        gaze_mapper.mapping_index_range = self._get_current_trim_mark_range()

    def set_validation_range_from_current_trim_marks(self, gaze_mapper):
        gaze_mapper.validation_index_range = self._get_current_trim_mark_range()

    def set_calibration_unique_id(self, gaze_mapper, calibration_unique_id):
        gaze_mapper.calibration_unique_id = calibration_unique_id

    def calculate(self, gaze_mapper):
        self._reset_gaze_mapper_results(gaze_mapper)
        calibration = self.get_valid_calibration_or_none(gaze_mapper)
        if calibration is None:
            self._abort_calculation(
                gaze_mapper,
                "The calibration was not found for the gaze mapper "
                f"'{gaze_mapper.name}', please select a different calibration!",
            )
            return None
        if calibration.params is None:
            self._abort_calculation(
                gaze_mapper,
                f"You first need to calculate calibration '{calibration.name}' before "
                f"calculating the mapper '{gaze_mapper.name}'",
            )
            return None
        try:
            task = self._create_mapping_task(gaze_mapper, calibration)
        except worker.map_gaze.NotEnoughPupilData:
            self._abort_calculation(gaze_mapper, "There is no pupil data to be mapped!")
            return None
        self._task_manager.add_task(task, identifier=f"{gaze_mapper.unique_id}-mapping")
        logger.info(f"Start gaze mapping for '{gaze_mapper.name}'")

    def _abort_calculation(self, gaze_mapper, error_message):
        logger.error(error_message)
        gaze_mapper.status = error_message
        self.on_calculation_could_not_be_started()
        # the gaze from this mapper got cleared, so don't show it anymore
        self.publish_all_enabled_mappers()

    def on_calculation_could_not_be_started(self):
        pass

    def _reset_gaze_mapper_results(self, gaze_mapper):
        gaze_mapper.gaze = []
        gaze_mapper.gaze_ts = []
        gaze_mapper.accuracy_result = ""
        gaze_mapper.precision_result = ""

    def _create_mapping_task(self, gaze_mapper, calibration):
        task = worker.map_gaze.create_task(gaze_mapper, calibration)

        def on_yield_gaze(mapped_gaze_ts_and_data):
            gaze_mapper.status = f"Mapping {task.progress * 100:.0f}% complete"
            for timestamp, gaze_datum in mapped_gaze_ts_and_data:
                gaze_mapper.gaze.append(gaze_datum)
                gaze_mapper.gaze_ts.append(timestamp)

        def on_completed_mapping(_):
            if gaze_mapper.empty():
                gaze_mapper.status = "No data mapped!"
                logger.warning(
                    f"Gaze mapper {gaze_mapper.name} produced no data."
                    f" Please check the quality of your Pupil data"
                    f" and ensure you are using the appropriate pipeline!"
                )
            else:
                gaze_mapper.status = "Successfully completed mapping"
            self.publish_all_enabled_mappers()
            self.validate_gaze_mapper(gaze_mapper)
            self._gaze_mapper_storage.save_to_disk()
            self.on_gaze_mapping_calculated(gaze_mapper)
            logger.info(f"Completed gaze mapping for '{gaze_mapper.name}'")

        task.add_observer("on_yield", on_yield_gaze)
        task.add_observer("on_completed", on_completed_mapping)
        task.add_observer("on_exception", tasklib.raise_exception)
        return task

    def publish_all_enabled_mappers(self):
        """
        Publish gaze data to e.g. render it in Player or to trigger other plugins
        that operate on gaze data. The publish logic is implemented in the plugin.
        """
        gaze_bisector = self._create_gaze_bisector_from_all_enabled_mappers()
        self._publish_gaze_bisector(gaze_bisector)

    def _create_gaze_bisector_from_all_enabled_mappers(self):
        gaze_data = list(
            chain.from_iterable(
                mapper.gaze
                for mapper in self._gaze_mapper_storage
                if mapper.activate_gaze
            )
        )
        gaze_ts = list(
            chain.from_iterable(
                mapper.gaze_ts
                for mapper in self._gaze_mapper_storage
                if mapper.activate_gaze
            )
        )
        return player_methods.Bisector(gaze_data, gaze_ts)

    def on_gaze_mapping_calculated(self, gaze_mapper):
        pass

    def validate_gaze_mapper(self, gaze_mapper):
        def validation_completed(accuracy_and_precision):
            accuracy, precision = accuracy_and_precision
            gaze_mapper.accuracy_result = (
                f"{accuracy.result:.1f}° from {accuracy.num_used} / "
                f"{accuracy.num_total} samples"
            )
            gaze_mapper.precision_result = (
                f"{precision.result:.1f}° from {precision.num_used} / "
                f"{precision.num_total} samples"
            )

        calibration = self.get_valid_calibration_or_none(gaze_mapper)

        if calibration is None:
            logger.error(
                f"Could not validate gaze mapper {gaze_mapper.name};"
                " Calibration was not found, please select a different calibration."
            )
            return

        if calibration.params is None:
            logger.error(
                f"Could not validate gaze mapper {gaze_mapper.name};"
                " You first need to calculate calibration '{calibration.name}'"
            )
            return

        task = worker.validate_gaze.create_bg_task(
            gaze_mapper, calibration, self._reference_location_storage
        )
        task.add_observer("on_completed", validation_completed)
        task.add_observer("on_exception", tasklib.raise_exception)
        self._task_manager.add_task(
            task, identifier=f"{gaze_mapper.unique_id}-validating"
        )

    def get_valid_calibration_or_none(self, gaze_mapper):
        return self._calibration_storage.get_or_none(gaze_mapper.calibration_unique_id)

    def on_gaze_mapper_deleted(self, *args, **kwargs):
        self.publish_all_enabled_mappers()
