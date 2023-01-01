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


class OfflineDetectionMenu:
    menu_label = "Marker Detection"

    def __init__(self, detection_controller, general_settings, index_range_as_str):
        self._detection_controller = detection_controller
        self._general_settings = general_settings
        self._index_range_as_str = index_range_as_str

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        detection_controller.add_observer(
            "on_detection_started", self._on_detection_started
        )
        detection_controller.add_observer(
            "on_detection_ended", self._on_detection_ended
        )

    def render(self):
        self.menu.elements.clear()
        self._render_ui()

    def _render_ui(self):
        self.menu.elements.extend(
            [self._create_range_selector(), self._create_toggle_detection_button()]
        )

    def _create_range_selector(self):
        range_string = "Detect markers in: " + self._index_range_as_str(
            self._general_settings.detection_frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set from trim marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_toggle_detection_button(self):
        if self._detection_controller.is_running_task:
            return ui.Button("Cancel detection", self._on_click_cancel_detection)
        else:
            return ui.Button("Start detection", self._on_click_start_detection)

    def _on_set_index_range_from_trim_marks(self):
        self._detection_controller.set_range_from_current_trim_marks()
        self.render()

    def _on_click_start_detection(self):
        self._detection_controller.calculate()

    def _on_click_cancel_detection(self):
        self._detection_controller.cancel_task()

    def _on_detection_started(self):
        self.render()

    def _on_detection_ended(self):
        self.render()
