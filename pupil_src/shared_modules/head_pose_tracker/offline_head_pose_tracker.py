"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import player_methods as pm
from head_pose_tracker import controller, storage
from head_pose_tracker import ui as plugin_ui
from plugin_timeline import PluginTimeline
from pupil_recording import PupilRecording
from tasklib.manager import PluginTaskManager

from .base_head_pose_tracker import Head_Pose_Tracker_Base


class Offline_Head_Pose_Tracker(Head_Pose_Tracker_Base):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the
    environment.
    """

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self._task_manager = PluginTaskManager(plugin=self)
        self._current_recording_uuid = str(
            PupilRecording(g_pool.rec_dir).meta_info.recording_uuid
        )

        self._setup_storages()
        self._setup_controllers()
        self._setup_renderers()
        self._setup_menus()
        self._setup_timelines()

    def _setup_storages(self):
        self._offline_settings_storage = storage.OfflineSettingsStorage(
            self.g_pool.rec_dir,
            plugin=self,
            get_recording_index_range=self._recording_index_range,
        )
        self._detection_storage = storage.OfflineDetectionStorage(
            self.g_pool.rec_dir,
            all_timestamps=self.g_pool.timestamps,
            plugin=self,
            get_current_frame_index=self.get_current_frame_index,
            get_current_frame_window=self.get_current_frame_window,
        )
        self._optimization_storage = storage.OptimizationStorage(
            self.g_pool.rec_dir,
            plugin=self,
            recording_uuid_current=self._current_recording_uuid,
        )
        self._localization_storage = storage.OfflineLocalizationStorage(
            self.g_pool.rec_dir,
            plugin=self,
            get_current_frame_index=self.get_current_frame_index,
            get_current_frame_window=self.get_current_frame_window,
        )

    def _setup_controllers(self):
        self._detection_controller = controller.OfflineDetectionController(
            self._offline_settings_storage,
            self._detection_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            all_timestamps=self.g_pool.timestamps,
            source_path=self.g_pool.capture.source_path,
        )
        self._optimization_controller = controller.OfflineOptimizationController(
            self._detection_controller,
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            self.g_pool.capture.intrinsics,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            all_timestamps=self.g_pool.timestamps,
            rec_dir=self.g_pool.rec_dir,
        )
        self._localization_controller = controller.OfflineLocalizationController(
            self._optimization_controller,
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self.g_pool.capture.intrinsics,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            all_timestamps=self.g_pool.timestamps,
        )
        self._export_controller = controller.ExportController(
            self._optimization_storage,
            self._localization_storage,
            task_manager=self._task_manager,
            plugin=self,
        )

    def _setup_renderers(self):
        self._detection_renderer = plugin_ui.DetectionRenderer(
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            plugin=self,
        )
        self._head_pose_tracker_3d_renderer = plugin_ui.HeadPoseTracker3DRenderer(
            self._offline_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self.g_pool.capture.intrinsics,
            plugin=self,
        )
        self._export_controller = controller.ExportController(
            self._optimization_storage,
            self._localization_storage,
            task_manager=self._task_manager,
            plugin=self,
        )

    def _setup_menus(self):
        self._visualization_menu = plugin_ui.VisualizationMenu(
            self._offline_settings_storage, self._head_pose_tracker_3d_renderer
        )
        self._detection_menu = plugin_ui.OfflineDetectionMenu(
            self._detection_controller,
            self._offline_settings_storage,
            index_range_as_str=self._index_range_as_str,
        )
        self._optimization_menu = plugin_ui.OfflineOptimizationMenu(
            self._optimization_controller,
            self._offline_settings_storage,
            self._optimization_storage,
            index_range_as_str=self._index_range_as_str,
        )
        self._localization_menu = plugin_ui.OfflineLocalizationMenu(
            self._localization_controller,
            self._offline_settings_storage,
            self._localization_storage,
            index_range_as_str=self._index_range_as_str,
        )
        self._head_pose_tracker_menu = plugin_ui.OfflineHeadPoseTrackerMenu(
            self._visualization_menu,
            self._detection_menu,
            self._optimization_menu,
            self._localization_menu,
            plugin=self,
        )

    def _setup_timelines(self):
        self._detection_timeline = plugin_ui.DetectionTimeline(
            self._detection_controller,
            self._offline_settings_storage,
            self._detection_storage,
            all_timestamps=self.g_pool.timestamps,
        )
        self._localization_timeline = plugin_ui.LocalizationTimeline(
            self._localization_controller,
            self._offline_settings_storage,
            self._localization_storage,
        )
        plugin_timeline = PluginTimeline(
            plugin=self,
            title="Offline Head Pose Tracker",
            timeline_ui_parent=self.g_pool.user_timelines,
            all_timestamps=self.g_pool.timestamps,
        )
        self._timeline = plugin_ui.OfflineHeadPoseTrackerTimeline(
            plugin_timeline,
            self._detection_timeline,
            self._localization_timeline,
            plugin=self,
        )

    def _recording_index_range(self):
        left_index = 0
        right_index = len(self.g_pool.timestamps) - 1
        return left_index, right_index

    def _current_trim_mark_range(self):
        right_idx = self.g_pool.seek_control.trim_right
        left_idx = self.g_pool.seek_control.trim_left
        return left_idx, right_idx

    def _index_range_as_str(self, index_range):
        from_index, to_index = index_range
        return "{} - {}".format(
            self._index_time_as_str(from_index), self._index_time_as_str(to_index)
        )

    def _index_time_as_str(self, index):
        ts = self.g_pool.timestamps[index]
        min_ts = self.g_pool.timestamps[0]
        time = ts - min_ts
        minutes = abs(time // 60)  # abs because it's sometimes -0
        seconds = round(time % 60)
        return f"{minutes:02.0f}:{seconds:02.0f}"

    def get_current_frame_index(self):
        return self.g_pool.capture.get_frame_index()

    def get_current_frame_window(self):
        frame_index = self.get_current_frame_index()
        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame_index)
        return frame_window
