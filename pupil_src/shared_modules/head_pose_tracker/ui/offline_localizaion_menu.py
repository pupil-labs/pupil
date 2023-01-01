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


class OfflineLocalizationMenu:
    menu_label = "Camera Localization"

    def __init__(
        self,
        localization_controller,
        general_settings,
        localization_storage,
        index_range_as_str,
    ):
        self._localization_controller = localization_controller
        self._general_settings = general_settings
        self._localization_storage = localization_storage

        self._index_range_as_str = index_range_as_str

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        localization_controller.add_observer(
            "on_localization_could_not_be_started",
            self._on_localization_could_not_be_started,
        )
        localization_controller.add_observer(
            "on_localization_ended", self._on_localization_ended
        )

    def render(self):
        self.menu.elements.clear()
        self._render_ui()

    def _render_ui(self):
        self.menu.elements.extend(
            [
                self._create_range_selector(),
                self._create_calculate_button(),
                self._create_status_text(),
            ]
        )

    def _create_range_selector(self):
        range_string = "Localize camera in: " + self._index_range_as_str(
            self._general_settings.localization_frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set from trim marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_calculate_button(self):
        return ui.Button(
            label="Recalculate"
            if self._localization_storage.calculated
            else "Calculate",
            function=self._on_click_calculate,
        )

    def _create_status_text(self):
        return ui.Text_Input(
            "status", self._localization_controller, label="Status", setter=lambda _: _
        )

    def _on_set_index_range_from_trim_marks(self):
        self._localization_controller.set_range_from_current_trim_marks()
        self.render()

    def _on_click_calculate(self):
        self._localization_controller.calculate()

    def _on_localization_could_not_be_started(self):
        self.render()

    def _on_localization_ended(self):
        self.render()
