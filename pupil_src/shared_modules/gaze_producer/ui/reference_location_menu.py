"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from pyglui import ui


class ReferenceLocationMenu:
    def __init__(self, reference_detection_controller, reference_location_storage,
                 reference_edit_controller):
        self._reference_detection_controller = reference_detection_controller
        self._reference_location_storage = reference_location_storage
        self._reference_edit_controller = reference_edit_controller
        self.menu = ui.Growing_Menu("Reference Locations")

    def render(self):
        self.menu.elements.clear()
        self._render_auto_detection_section()
        self.menu.append(ui.Separator())
        self._render_manual_edit_section()
        self.menu.append(ui.Separator())
        self._render_clear_all_section()

    def _render_auto_detection_section(self):
        self.menu.append(self._make_button_toggle_reference_detection())

    def _make_button_toggle_reference_detection(self):
        if self._reference_detection_controller.is_running_detection:
            return ui.Button(
                "Cancel Detection", self._on_click_cancel_reference_detection
            )
        else:
            return ui.Button(
                "Detect Circle Markers in Recording",
                self._on_click_start_reference_detection,
            )

    def _render_manual_edit_section(self):
        self.menu.append(
            ui.Switch(
                "edit_mode_active",
                self._reference_edit_controller,
                label="Manual Edit Mode",
            )
        )
        self.menu.append(
            ui.Button(
                outer_label="Jump to",
                label="Next Reference Location",
                function=self._on_jump_to_next_ref,
            )
        )
        self.menu.append(
            ui.Button(
                outer_label="Jump to",
                label="Previous Reference Location",
                function=self._on_jump_to_prev_ref,
            )
        )

    def _render_clear_all_section(self):
        self.menu.append(
            ui.Button("Clear All Reference Locations", self._on_clear_all_refs)
        )

    def _on_click_start_reference_detection(self):
        task = self._reference_detection_controller.start_detection()
        task.add_observer("on_started", self._on_started_reference_detection)
        task.add_observer("on_ended", self._on_ended_reference_detection)

    def _on_click_cancel_reference_detection(self):
        self._reference_detection_controller.cancel_detection()

    def _on_started_reference_detection(self):
        self.render()

    def _on_ended_reference_detection(self):
        self.render()

    def _on_jump_to_next_ref(self):
        self._reference_edit_controller.jump_to_next_ref()

    def _on_jump_to_prev_ref(self):
        self._reference_edit_controller.jump_to_prev_ref()

    def _on_clear_all_refs(self):
        self._reference_location_storage.delete_all()
