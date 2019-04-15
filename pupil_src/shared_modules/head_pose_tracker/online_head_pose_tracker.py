"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from head_pose_tracker import ui as plugin_ui, controller, storage

from observable import Observable
from plugin import Plugin
from tasklib.manager import PluginTaskManager


class Online_Head_Pose_Tracker(Plugin, Observable):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the
    environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self._task_manager = PluginTaskManager(plugin=self)

        self._setup_storages()
        self._setup_controllers()
        self._setup_renderers()
        self._setup_menus()

    def _setup_storages(self):
        self._online_settings_storage = storage.OnlineSettingsStorage(
            self.g_pool.user_dir, plugin=self
        )
        self._detection_storage = storage.OnlineDetectionStorage()
        self._optimization_storage = storage.OptimizationStorage(
            self.g_pool.user_dir, plugin=self
        )
        self._localization_storage = storage.OnlineLocalizationStorage()

    def _setup_controllers(self):
        self._controller = controller.OnlineController(
            self._online_settings_storage,
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
            self._online_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            plugin=self,
        )
        self._head_pose_tracker_3d_renderer = plugin_ui.HeadPoseTracker3DRenderer(
            self._online_settings_storage,
            self._detection_storage,
            self._optimization_storage,
            self._localization_storage,
            self.g_pool.capture.intrinsics,
            plugin=self,
        )

    def _setup_menus(self):
        self._optimization_menu = plugin_ui.OnlineOptimizationMenu(
            self._controller, self._online_settings_storage, self._optimization_storage
        )
        self._localization_menu = plugin_ui.OnlineLocalizationMenu(
            self._online_settings_storage, self._localization_storage
        )
        self._head_pose_tracker_menu = plugin_ui.OnlineHeadPoseTrackerMenu(
            self._optimization_menu,
            self._localization_menu,
            self._head_pose_tracker_3d_renderer,
            plugin=self,
        )
