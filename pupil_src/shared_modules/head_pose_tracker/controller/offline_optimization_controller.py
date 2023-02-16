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


class OfflineOptimizationController(Observable):
    def __init__(
        self,
        detection_controller,
        general_settings,
        detection_storage,
        optimization_storage,
        camera_intrinsics,
        task_manager,
        get_current_trim_mark_range,
        all_timestamps,
        rec_dir,
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._get_current_trim_mark_range = get_current_trim_mark_range
        self._all_timestamps = all_timestamps
        self._rec_dir = rec_dir

        self._task = None

        if self._optimization_storage.calculated:
            self.status = "calculated"
        else:
            self.status = self._default_status

        detection_controller.add_observer(
            "on_detection_ended", self._on_detection_ended
        )

    @property
    def _default_status(self):
        return "Not calculated yet"

    def _on_detection_ended(self):
        if (
            self._optimization_storage.is_from_same_recording
            and not self._optimization_storage.calculated
        ):
            self.calculate()
        else:
            self.on_optimization_had_completed_before()

    def calculate(self):
        self._reset()
        self._create_optimization_task()

    def _reset(self):
        self.cancel_task()
        self._optimization_storage.set_to_default_values()
        self.status = self._default_status

    def _create_optimization_task(self):
        def on_yield(result):
            self._update_result(result)
            self.status = f"{self._task.progress * 100:.0f}% completed"

        def on_completed(_):
            if self._optimization_storage.calculated:
                self._camera_intrinsics.save(self._rec_dir)
                self.status = "successfully completed"
                self.on_optimization_completed()
            else:
                if self._general_settings.user_defined_origin_marker_id is not None:
                    reason = (
                        "not enough markers with the defined origin marker id "
                        "were collected"
                    )
                else:
                    reason = "not enough markers were collected"

                self.status = "failed: " + reason
            logger.info(f"markers 3d model optimization '{self.status}' ")

            self._optimization_storage.save_plmodel_to_disk()

        self._task = self._create_task()
        self._task.add_observer("on_yield", on_yield)
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_exception", tasklib.raise_exception)
        self._task.add_observer("on_started", self.on_optimization_started)
        logger.info("Start markers 3d model optimization")
        self.status = "0% completed"

    def _create_task(self):
        args = (
            self._all_timestamps,
            self._general_settings.optimization_frame_index_range,
            self._general_settings.user_defined_origin_marker_id,
            self._general_settings.optimize_camera_intrinsics,
            self._detection_storage.markers_bisector,
            self._detection_storage.frame_index_to_num_markers,
            self._camera_intrinsics,
        )
        return self._task_manager.create_background_task(
            name="markers 3d model optimization",
            routine_or_generator_function=worker.offline_optimization,
            pass_shared_memory=True,
            args=args,
        )

    def _update_result(self, result):
        model_tuple, intrinsics_tuple = result
        self._optimization_storage.update_model(*model_tuple)
        self._camera_intrinsics.update_camera_matrix(intrinsics_tuple.camera_matrix)
        self._camera_intrinsics.update_dist_coefs(intrinsics_tuple.dist_coefs)

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    def set_range_from_current_trim_marks(self):
        self._general_settings.optimization_frame_index_range = (
            self._get_current_trim_mark_range()
        )

    def on_optimization_had_completed_before(self):
        pass

    def on_optimization_started(self):
        pass

    def on_optimization_completed(self):
        pass
