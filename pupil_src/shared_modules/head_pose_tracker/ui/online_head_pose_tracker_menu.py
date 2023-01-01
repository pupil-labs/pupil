"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from pyglui import ui


class OnlineHeadPoseTrackerMenu:
    def __init__(self, visualization_menu, optimization_menu, plugin):
        self._visualization_menu = visualization_menu
        self._optimization_menu = optimization_menu
        self._plugin = plugin

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)

    def _on_init_ui(self):
        self._plugin.add_menu()
        self._plugin.menu.label = "Online Head Pose Tracker"

        self._plugin.menu.extend(self._render_on_top_menu())

        self._optimization_menu.render()
        self._plugin.menu.append(self._optimization_menu.menu)

        self._visualization_menu.render()
        self._plugin.menu.append(self._visualization_menu.menu)

    def _on_deinit_ui(self):
        self._plugin.remove_menu()

    def _render_on_top_menu(self):
        menu = [self._create_on_top_text()]
        return menu

    def _create_on_top_text(self):
        return ui.Info_Text(
            "This plugin allows you to track camera poses in relation to the "
            "printed markers in the scene. \n "
            "First, marker are detected. "
            "Second, based on the detections, markers 3d model is built. "
            "Third, camera localizations is calculated."
        )


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
        self._render_ui()

    def _render_ui(self):
        self.menu.elements.extend(
            [
                self._create_name_input(),
                self._create_optimize_markers_3d_model_switch(),
                self._create_optimize_camera_intrinsics_switch(),
                self._create_origin_marker_id_display(),
                self._create_reset_markers_3d_model_button(),
            ]
        )

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
            label="Build markers 3d model",
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
            label="Origin marker id",
            getter=self._on_get_origin_marker_id,
            setter=lambda _: _,
        )

    def _create_reset_markers_3d_model_button(self):
        return ui.Button("Reset", function=self._controller.reset)

    def _on_name_change(self, new_name):
        self._optimization_storage.rename(new_name)
        self.render()

    def _on_get_origin_marker_id(self):
        origin_marker_id = self._optimization_storage.origin_marker_id
        return str(origin_marker_id)

    def _on_optimize_markers_3d_model_switched(self, new_value):
        self._controller.switch_optimize_markers_3d_model(new_value)
        self.render()
