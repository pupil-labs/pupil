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


class ReferenceLocationMenu:
    def __init__(
        self,
        reference_detection_controller,
        reference_location_storage,
        reference_edit_controller,
    ):
        self._reference_detection_controller = reference_detection_controller
        self._reference_location_storage = reference_location_storage
        self._reference_edit_controller = reference_edit_controller
        self.menu = ui.Growing_Menu("Reference Locations")
        self.menu.collapsed = True

        reference_detection_controller.add_observer(
            "on_detection_started", self._on_started_reference_detection
        )

    def render(self):
        self.menu.elements.clear()
        self.menu.extend(
            [
                self._create_toggle_reference_detection_button(),
                ui.Separator(),
                self._create_edit_mode_switch(),
                self._create_edit_mode_explanation(),
                self._create_next_ref_button(),
                self._create_previous_ref_button(),
                ui.Separator(),
                self._create_clear_all_button(),
            ]
        )

    def _create_toggle_reference_detection_button(self):
        if self._reference_detection_controller.is_running_detection:
            return ui.Button(
                "Cancel Detection", self._on_click_cancel_reference_detection
            )
        else:
            return ui.Button(
                "Detect Circle Markers in Recording",
                self._on_click_start_reference_detection,
            )

    def _create_edit_mode_switch(self):
        return ui.Switch(
            "edit_mode_active",
            self._reference_edit_controller,
            label="Manual Edit Mode",
        )

    def _create_edit_mode_explanation(self):
        return ui.Info_Text(
            "When the edit mode is active, click on the video to set "
            "the reference for the current frame. Click on a "
            "reference to delete it."
        )

    def _create_next_ref_button(self):
        return ui.Button(
            outer_label="Jump to",
            label="Next Reference Location",
            function=self._on_jump_to_next_ref,
        )

    def _create_previous_ref_button(self):
        return ui.Button(
            outer_label="Jump to",
            label="Previous Reference Location",
            function=self._on_jump_to_prev_ref,
        )

    def _create_clear_all_button(self):
        return ui.Button("Clear All Reference Locations", self._on_clear_all_refs)

    def _on_click_start_reference_detection(self):
        self._reference_detection_controller.start_detection()

    def _on_click_cancel_reference_detection(self):
        self._reference_detection_controller.cancel_detection()

    def _on_started_reference_detection(self, detection_task):
        detection_task.add_observer("on_ended", self._on_ended_reference_detection)
        self.render()

    def _on_ended_reference_detection(self):
        self.render()

    def _on_jump_to_next_ref(self):
        self._reference_edit_controller.jump_to_next_ref()

    def _on_jump_to_prev_ref(self):
        self._reference_edit_controller.jump_to_prev_ref()

    def _on_clear_all_refs(self):
        self._reference_location_storage.delete_all()
