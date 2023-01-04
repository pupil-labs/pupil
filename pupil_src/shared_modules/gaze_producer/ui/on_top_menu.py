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


class OnTopMenu:
    """The part of the menu that's above all other menus (reference locations etc.)"""

    def __init__(self, calculate_all_controller, reference_location_storage):
        self._calculate_all_button = None

        self._calculate_all_controller = calculate_all_controller
        self._reference_location_storage = reference_location_storage

        reference_location_storage.add_observer(
            "add", self._on_reference_storage_changed
        )
        reference_location_storage.add_observer(
            "delete", self._on_reference_storage_changed
        )
        reference_location_storage.add_observer(
            "delete_all", self._on_reference_storage_changed
        )

    def render(self, menu):
        self._calculate_all_button = self._create_calculate_all_button()
        menu.extend(
            [
                ui.Info_Text(
                    "This plugin allows you to create gaze based on custom "
                    "calibrations. First, you specify or search automatically for "
                    "reference locations. Using these, you create one or more "
                    "calibrations. Third, you create one or more gaze mappers."
                ),
                ui.Info_Text(
                    "You can perform all steps individually or click the button below, "
                    "which performs all steps with the current settings."
                ),
                self._calculate_all_button,
            ]
        )

    def _create_calculate_all_button(self):
        return ui.Button(
            label=self._calculate_all_button_label,
            function=self._calculate_all_controller.calculate_all,
        )

    @property
    def _calculate_all_button_label(self):
        if self._reference_location_storage.is_empty:
            return "Detect References, Calculate All Calibrations and Mappings"
        else:
            return "Calculate All Calibrations and Mappings"

    def _on_reference_storage_changed(self, *args, **kwargs):
        self._calculate_all_button.label = self._calculate_all_button_label
