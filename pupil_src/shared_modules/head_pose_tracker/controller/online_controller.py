"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import tasklib
from head_pose_tracker import worker
from head_pose_tracker.function import pick_key_markers


class OnlineController:
    def __init__(
        self,
        general_settings,
        detection_storage,
        optimization_storage,
        localization_storage,
        camera_intrinsics,
        task_manager,
        user_dir,
        plugin,
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._camera_intrinsics = camera_intrinsics
        self._task_manager = task_manager
        self._user_dir = user_dir

        self._task = None

        # first trigger
        self._calculate_markers_3d_model()

        plugin.add_observer("recent_events", self._on_recent_events)
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_recent_events(self, events):
        if "frame" in events:
            self._calculate_current_markers(events["frame"])
            self._calculate_current_pose(events["frame"])
            self._save_key_markers()
            events["head_pose"] = self._create_head_pose_events()

    def _calculate_current_markers(self, frame):
        self._detection_storage.current_markers = worker.online_detection(frame)

    def _calculate_current_pose(self, frame):
        self._localization_storage.current_pose = worker.online_localization(
            frame.timestamp,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self._camera_intrinsics,
        )

    def _create_head_pose_events(self):
        """
        Creates head pose events to be added to the current list of events.
        """
        position = {"topic": "head_pose"}
        position.update(self._localization_storage.current_pose)
        return [position]

    def _save_key_markers(self):
        if self._general_settings.optimize_markers_3d_model:
            self._optimization_storage.all_key_markers += pick_key_markers.run(
                self._detection_storage.current_markers,
                self._optimization_storage.all_key_markers,
            )

    def _calculate_markers_3d_model(self):
        if (
            not self.is_running_task
            and self._general_settings.optimize_markers_3d_model
        ):
            self._create_optimization_task()

    def _create_optimization_task(self):
        def on_completed(result):
            self._update_result(result)

            # Start again when the task is done
            self._calculate_markers_3d_model()

        self._task = self._create_task()
        self._task.add_observer("on_completed", on_completed)
        self._task.add_observer("on_exception", tasklib.raise_exception)

    def _create_task(self):
        args = (
            self._optimization_storage.origin_marker_id,
            self._optimization_storage.marker_id_to_extrinsics,
            self._optimization_storage.frame_id_to_extrinsics,
            self._optimization_storage.all_key_markers,
            self._general_settings.optimize_camera_intrinsics,
            self._camera_intrinsics,
        )
        return self._task_manager.create_background_task(
            name="markers 3d model optimization",
            routine_or_generator_function=worker.online_optimization,
            pass_shared_memory=False,
            args=args,
        )

    def _update_result(self, result):
        if not result:
            return

        model_tuple, frame_id_to_extrinsics, frame_ids_failed, intrinsics_tuple = result
        self._optimization_storage.update_model(*model_tuple)
        self._optimization_storage.frame_id_to_extrinsics = frame_id_to_extrinsics
        self._optimization_storage.discard_failed_key_markers(frame_ids_failed)
        self._optimization_storage.save_plmodel_to_disk()

        self._camera_intrinsics.update_camera_matrix(intrinsics_tuple.camera_matrix)
        self._camera_intrinsics.update_dist_coefs(intrinsics_tuple.dist_coefs)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    def cancel_task(self):
        if self.is_running_task:
            self._task.kill(None)

    def _on_cleanup(self):
        self._optimization_storage.save_plmodel_to_disk()
        self._camera_intrinsics.save(self._user_dir)

    def reset(self):
        self.cancel_task()
        self._optimization_storage.set_to_default_values()
        self._localization_storage.set_to_default_values()
        self._calculate_markers_3d_model()

    def switch_optimize_markers_3d_model(self, new_value):
        self._general_settings.optimize_markers_3d_model = new_value
        if new_value:
            self._calculate_markers_3d_model()
        else:
            self.cancel_task()
