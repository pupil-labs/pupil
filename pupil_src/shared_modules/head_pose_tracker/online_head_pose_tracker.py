"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import controller, storage
from head_pose_tracker import ui as plugin_ui
from tasklib.manager import PluginTaskManager

from .base_head_pose_tracker import Head_Pose_Tracker_Base


class Online_Head_Pose_Tracker(Head_Pose_Tracker_Base):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the
    environment.
    """

    def __init__(
        self,
        g_pool,
        optimize_markers_3d_model=False,
        optimize_camera_intrinsics=True,
        open_visualization_window=False,
        show_camera_trace_in_3d_window=False,
        render_markers=True,
        show_marker_id_in_main_window=False,
        window_size=(1000, 1000),
        window_position=(90, 90),
    ):
        super().__init__(g_pool)

        self._task_manager = PluginTaskManager(plugin=self)

        self._online_settings = storage.OnlineSettings(
            (
                optimize_markers_3d_model,
                optimize_camera_intrinsics,
                open_visualization_window,
                show_camera_trace_in_3d_window,
                render_markers,
                show_marker_id_in_main_window,
                window_size,
                window_position,
            )
        )
        self._setup_storages()
        self._setup_controllers()
        self._setup_renderers()
        self._setup_menus()

    def _setup_storages(self):
        self._detection_storage = storage.OnlineDetectionStorage()
        self._optimization_storage = storage.OptimizationStorage(
            self.g_pool.user_dir, plugin=self
        )
        self._localization_storage = storage.OnlineLocalizationStorage()

    def _setup_controllers(self):
        self._controller = controller.OnlineController(
            self._online_settings,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self.g_pool.capture.intrinsics,
            self._task_manager,
            self.g_pool.user_dir,
            plugin=self,
        )

    def _setup_renderers(self):
        self._detection_renderer = plugin_ui.DetectionRenderer(
            self._online_settings,
            self._detection_storage,
            self._optimization_storage,
            plugin=self,
        )
        self._head_pose_tracker_3d_renderer = plugin_ui.HeadPoseTracker3DRenderer(
            self._online_settings,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self.g_pool.capture.intrinsics,
            plugin=self,
        )

    def _setup_menus(self):
        self._visualization_menu = plugin_ui.VisualizationMenu(
            self._online_settings, self._head_pose_tracker_3d_renderer
        )
        self._optimization_menu = plugin_ui.OnlineOptimizationMenu(
            self._controller, self._online_settings, self._optimization_storage
        )
        self._head_pose_tracker_menu = plugin_ui.OnlineHeadPoseTrackerMenu(
            self._visualization_menu, self._optimization_menu, plugin=self
        )

    def get_init_dict(self):
        return self._online_settings.data_as_dict
