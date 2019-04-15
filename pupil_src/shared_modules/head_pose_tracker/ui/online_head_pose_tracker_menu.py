"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from pyglui import ui


class OnlineHeadPoseTrackerMenu:
    def __init__(
        self,
        optimization_menu,
        localization_menu,
        head_pose_tracker_3d_renderer,
        plugin,
    ):
        self._optimization_menu = optimization_menu
        self._localization_menu = localization_menu
        self._head_pose_tracker_3d_renderer = head_pose_tracker_3d_renderer
        self._plugin = plugin

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Online Head Pose Tracker"

        self._plugin.menu.extend(self._render_on_top_menu())

        self._optimization_menu.render()
        self._plugin.menu.append(self._optimization_menu.menu)

        self._localization_menu.render()
        self._plugin.menu.append(self._localization_menu.menu)

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _render_on_top_menu(self):
        menu = [
            self._create_on_top_text(),
            self._create_open_visualization_window_switch(),
        ]
        return menu

    def _create_on_top_text(self):
        return ui.Info_Text(
            "This plugin allows you to track camera poses in relation to the "
            "printed markers in the scene. \n "
            "First, marker are detected. "
            "Second, based on the detections, markers 3d model is built. "
            "Third, camera localizations is calculated."
        )

    def _create_open_visualization_window_switch(self):
        return ui.Switch(
            "open_visualization_window",
            self._head_pose_tracker_3d_renderer,
            label="Open Visualization Window",
            setter=self._on_open_visualization_window_switched,
        )

    def _on_open_visualization_window_switched(self, new_value):
        self._head_pose_tracker_3d_renderer.switch_visualization_window(new_value)


class OnlineOptimizationMenu:
    menu_label = "Markers 3D Model"

    def __init__(self, controller, general_settings, optimization_storage):
        self._controller = controller
        self._general_settings = general_settings
        self._optimization_storage = optimization_storage

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        self.menu.elements.extend(self._render_ui())

    def _render_ui(self):
        menu = [
            self._create_name_input(),
            self._create_optimize_markers_3d_model_switch(),
            self._create_optimize_camera_intrinsics_switch(),
            self._create_origin_marker_id_display(),
            self._create_show_marker_id_switch(),
            self._create_reset_markers_3d_model_button(),
        ]
        return menu

    def _create_name_input(self):
        return ui.Text_Input(
            "name",
            self._optimization_storage,
            label="Name",
            setter=self._on_name_change,
        )

    def _create_optimize_markers_3d_model_switch(self):
        return ui.Switch(
            "optimize_markers_3d_model",
            self._general_settings,
            label="Build Markers 3D Model",
            setter=self._on_optimize_markers_3d_model_switched,
        )

    def _create_optimize_camera_intrinsics_switch(self):
        switch = ui.Switch(
            "optimize_camera_intrinsics",
            self._general_settings,
            label="Optimize camera intrinsics",
        )
        if not self._general_settings.optimize_markers_3d_model:
            switch.read_only = True
        return switch

    def _create_origin_marker_id_display(self):
        return ui.Text_Input(
            "origin_marker_id",
            self._optimization_storage,
            label="origin marker id",
            getter=self._on_get_origin_marker_id,
            setter=lambda _: _,
        )

    def _create_show_marker_id_switch(self):
        return ui.Switch(
            "show_marker_id", self._general_settings, label="Show Marker IDs"
        )

    def _create_reset_markers_3d_model_button(self):
        return ui.Button("reset", function=self._controller.reset)

    def _on_name_change(self, new_name):
        self._optimization_storage.rename(new_name)
        self.render()

    def _on_get_origin_marker_id(self):
        origin_marker_id = self._optimization_storage.origin_marker_id
        return str(origin_marker_id)

    def _on_optimize_markers_3d_model_switched(self, new_value):
        self._controller.switch_optimize_markers_3d_model(new_value)
        self.render()


class OnlineLocalizationMenu:
    menu_label = "Camera Localizer"

    def __init__(self, general_settings, localization_storage):
        self._general_settings = general_settings
        self._localization_storage = localization_storage

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

    def render(self):
        self.menu.elements.clear()
        self._render_custom_ui()

    def _render_custom_ui(self):
        self.menu.elements.extend([self._create_show_camera_trace_switch()])

    def _create_show_camera_trace_switch(self):
        return ui.Switch(
            "show_camera_trace", self._general_settings, label="Show Camera Trace"
        )
