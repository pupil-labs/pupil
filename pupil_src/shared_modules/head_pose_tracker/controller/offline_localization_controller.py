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
from head_pose_tracker import worker
from observable import Observable

logger = logging.getLogger(__name__)


class OfflineLocalizationController(Observable):
    def __init__(
        self,
        optimization_controller,
        general_settings,
        detection_storage,
        optimization_storage,
        localization_storage,
        camera_intrinsics,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps

        self._task = None

        if self._localization_storage.calculated:
            self.status = "calculated"
        else:
            self.status = self._default_status

        optimization_controller.add_observer(
            "on_optimization_had_completed_before",
            self._on_optimization_had_completed_before,
        )
        optimization_controller.add_observer(
            "on_optimization_started", self._on_optimization_started
        )
        optimization_controller.add_observer(
            "on_optimization_completed", self._on_optimization_completed
        )

    @property
    def _default_status(self):
        return "Not calculated yet"

    def _on_optimization_had_completed_before(self):
        if not self._localization_storage.calculated:
            self.calculate()

    def _on_optimization_started(self):
        self.reset()

    def _on_optimization_completed(self):
        self.calculate()

    def calculate(self):
        if not self._check_valid_markers_3d_model():
            return

        self.reset()
        self._create_localization_task()

    def _check_valid_markers_3d_model(self):
        if not self._optimization_storage.calculated:
            error_message = (
                "failed: markers 3d model '{}' should be calculated before calculating"
                " camera localization".format(self._optimization_storage.name)
            )
            self._abort_calculation(error_message)
            return False
        return True

    def _abort_calculation(self, error_message):
        logger.error(error_message)
        self.status = error_message
        self.on_localization_could_not_be_started()

    def reset(self):
        self.cancel_task()
        self._localization_storage.set_to_default_values()
        self.status = self._default_status

    def _create_localization_task(self):
        def on_yield(data_pairs):
            self._insert_pose_bisector(data_pairs)
            self.status = f"{self._task.progress * 100:.0f}% completed"

        def on_completed(_):
            self.status = "successfully completed"
            self._localization_storage.save_pldata_to_disk()
            logger.info("camera localization completed")
            self.on_localization_ended()

        def on_canceled_or_killed():
            self._localization_storage.save_pldata_to_disk()
            logger.info("camera localization canceled")
            self.on_localization_ended()

        self._task = self._create_task()
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_canceled_or_killed", on_canceled_or_killed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_localization_started)
        logger.info("Start camera localization")
        self.status = "0% completed"

    def _create_task(self):
        args = (
            self._all_timestamps,
            self._general_settings.localization_frame_index_range,
            self._detection_storage.markers_bisector,
            self._detection_storage.frame_index_to_num_markers,
            self._optimization_storage.marker_id_to_extrinsics,
            self._camera_intrinsics,
        )
        return self._task_manager.create_background_task(
            name="camera localization",
            routine_or_generator_function=worker.offline_localization,
            pass_shared_memory=True,
            args=args,
        )

    def _insert_pose_bisector(self, data_pairs):
        for timestamp, pose in data_pairs:
            self._localization_storage.pose_bisector.insert(timestamp, pose)
        self.on_localization_yield()

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    @property
    def progress(self):
        return self._task.progress if self.is_running_task else 0.0

    def set_range_from_current_trim_marks(self):
        self._general_settings.localization_frame_index_range = (
            self._get_current_trim_mark_range()
        )

    def on_localization_could_not_be_started(self):
        pass

    def on_localization_started(self):
        pass

    def on_localization_yield(self):
        pass

    def on_localization_ended(self):
        pass
