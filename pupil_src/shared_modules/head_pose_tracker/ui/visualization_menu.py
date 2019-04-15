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


class VisualizationMenu:
    menu_label = "Visualization options"

    def __init__(self, general_settings, head_pose_tracker_3d_renderer):
        self._general_settings = general_settings
        self._head_pose_tracker_3d_renderer = head_pose_tracker_3d_renderer

        head_pose_tracker_3d_renderer.add_observer(
            "switch_visualization_window", self._on_switch_visualization_window
        )

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        self._spaces = "        "

    def render(self):
        self.menu.elements.clear()
        self._render_ui()

    def _render_ui(self):
        menu = [
            self._create_color_info_text(),
            self._create_render_markers_switch(),
            self._create_show_marker_id_in_main_window_switch(),
            ui.Separator(),
            self._create_open_visualization_window_switch(),
            self._create_show_camera_trace_switch(),
            self._create_show_marker_id_in_3d_window_switch(),
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
            label=self._spaces + "Render markers",
            setter=self._on_render_markers_switched,
        )

    def _create_show_marker_id_in_main_window_switch(self):
        switch = ui.Switch(
            "show_marker_id_in_main_window",
            self._general_settings,
            label=self._spaces + "Show marker id",
        )
        if not self._general_settings.render_markers:
            switch.read_only = True
        return switch

    def _create_open_visualization_window_switch(self):
        Button = ui.Button(
            outer_label="Visualization 3d window",
            label="Open",
            function=self._on_open_visualization_window_button_clicked,
        )
        if self._general_settings.open_visualization_window:
            Button.read_only = True
        return Button

    def _create_show_camera_trace_switch(self):
        switch = ui.Switch(
            "show_camera_trace_in_3d_window",
            self._general_settings,
            label=self._spaces + "Show camera trace",
        )
        if not self._general_settings.open_visualization_window:
            switch.read_only = True
        return switch

    def _create_show_marker_id_in_3d_window_switch(self):
        switch = ui.Switch(
            "show_marker_id_in_3d_window",
            self._general_settings,
            label=self._spaces + "Show marker id",
        )
        if not self._general_settings.open_visualization_window:
            switch.read_only = True
        return switch

    def _on_render_markers_switched(self, new_value):
        self._general_settings.render_markers = new_value
        self.render()

    def _on_open_visualization_window_button_clicked(self):
        self._head_pose_tracker_3d_renderer.switch_visualization_window(True)

    def _on_switch_visualization_window(self, _):
        self.render()
