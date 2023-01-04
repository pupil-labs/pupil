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

from head_pose_tracker import worker
from observable import Observable

logger = logging.getLogger(__name__)


class ExportController(Observable):
    def __init__(
        self, optimization_storage, localization_storage, task_manager, plugin
    ):
        self._task = None
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._task_manager = task_manager
        plugin.add_observer("on_notify", self._on_notify)
        plugin.add_observer("cleanup", self.cancel_export)

    @property
    def is_running_task(self):
        return self._task is not None and self._task.running

    def cancel_export(self):
        if self.is_running_task:
            self._task.kill(None)

    def _on_notify(self, notification):
        if notification["subject"] == "should_export":
            self._on_should_export(
                notification["export_dir"], notification["ts_window"]
            )

    def _on_should_export(self, export_dir, export_window):
        model_flat = self._3d_model_as_list()
        poses = self._camera_poses(export_window)
        self._task = self._create_export_task(export_dir, model_flat, poses)

    def _create_export_task(self, rec_dir, model, poses):
        args = (rec_dir, model, poses)
        return self._task_manager.create_background_task(
            name="head pose data export",
            routine_or_generator_function=worker.export_routine,
            pass_shared_memory=False,
            args=args,
        )

    def _3d_model_as_list(self):
        return self._optimization_storage.flattened_vertices()

    def _camera_poses(self, ts_window):
        return self._localization_storage.pose_bisector.by_ts_window(ts_window)
