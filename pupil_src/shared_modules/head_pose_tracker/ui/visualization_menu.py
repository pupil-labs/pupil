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

_SPACES = "        "


class VisualizationMenu:
    menu_label = "Visualization options"

    def __init__(self, general_settings, head_pose_tracker_3d_renderer):
        self._general_settings = general_settings
        self._head_pose_tracker_3d_renderer = head_pose_tracker_3d_renderer

        head_pose_tracker_3d_renderer.add_observer(
            "toggle_visualization_window", self._on_toggle_visualization_window
        )

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

    def render(self):
        self.menu.elements.clear()
        self._render_ui()

    def _render_ui(self):
        menu = [
            self._create_color_info_text(),
            self._create_render_markers_switch(),
            self._create_show_marker_id_in_main_window_switch(),
            ui.Separator(),
            self._create_toggle_visualization_window_button(),
            self._create_show_camera_trace_switch(),
        ]
        self.menu.elements.extend(menu)

    def _create_color_info_text(self):
        return ui.Info_Text(
            "Markers in current frame are drawn red in main window "
            "if they are part of the markers of 3d model; "
            "Otherwise, they are drawn green"
        )

    def _create_render_markers_switch(self):
        return ui.Switch(
            "render_markers",
            self._general_settings,
            label=_SPACES + "Render markers",
            setter=self._on_render_markers_switched,
        )

    def _create_show_marker_id_in_main_window_switch(self):
        switch = ui.Switch(
            "show_marker_id_in_main_window",
            self._general_settings,
            label=_SPACES + "Show marker id",
        )
        if not self._general_settings.render_markers:
            switch.read_only = True
        return switch

    def _create_toggle_visualization_window_button(self):
        label = "Close" if self._general_settings.open_visualization_window else "Open"
        button = ui.Button(
            outer_label="Visualization 3d window",
            label=label,
            function=self._on_toggle_visualization_window_button_clicked,
        )
        return button

    def _create_show_camera_trace_switch(self):
        switch = ui.Switch(
            "show_camera_trace_in_3d_window",
            self._general_settings,
            label=_SPACES + "Show camera trace",
        )
        if not self._general_settings.open_visualization_window:
            switch.read_only = True
        return switch

    def _on_render_markers_switched(self, new_value):
        self._general_settings.render_markers = new_value
        self.render()

    def _on_toggle_visualization_window_button_clicked(self):
        self._head_pose_tracker_3d_renderer.toggle_visualization_window()

    def _on_toggle_visualization_window(self):
        self.render()
