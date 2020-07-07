"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from pyglui import ui

from gaze_producer import ui as plugin_ui
from gaze_mapping import (
    gazer_labels_by_class_names,
    registered_gazer_classes,
    user_selectable_gazer_classes,
)

logger = logging.getLogger(__name__)


class CalibrationMenu(plugin_ui.StorageEditMenu):
    menu_label = "Calibrations"
    selector_label = "Edit Calibration:"
    new_button_label = "New Calibration"
    duplicate_button_label = "Duplicate Current Calibration"

    def __init__(self, calibration_storage, calibration_controller, index_range_as_str):
        super().__init__(calibration_storage)
        self._calibration_storage = calibration_storage
        self._calibration_controller = calibration_controller
        self._index_range_as_str = index_range_as_str

        self.menu.collapsed = True

        calibration_controller.add_observer(
            "on_calibration_computed", self._on_calibration_computed
        )
        calibration_controller.add_observer(
            "on_calculation_could_not_be_started",
            self._on_calculation_could_not_be_started,
        )

    def _item_label(self, calibration):
        return calibration.name

    def _new_item(self):
        return self._calibration_storage.create_default_calibration()

    def _duplicate_item(self, calibration):
        return self._calibration_storage.duplicate_calibration(calibration)

    def _render_custom_ui(self, calibration, menu):
        if not self._calibration_controller.is_from_same_recording(calibration):
            self._render_ui_calibration_from_other_recording(calibration, menu)
        elif not calibration.is_offline_calibration:
            self._render_ui_online_calibration(calibration, menu)
        else:
            self._render_ui_normally(calibration, menu)

    def _render_ui_normally(self, calibration, menu):
        menu.extend(
            [
                self._create_name_input(calibration),
                self._create_range_selector(calibration),
                self._create_mapping_method_selector(calibration),
                self._create_min_confidence_slider(calibration),
                self._create_status_display(calibration),
                self._create_calculate_button(calibration),
            ]
        )

    def _create_name_input(self, calibration):
        return ui.Text_Input(
            "name", calibration, label="Name", setter=self._on_name_change
        )

    def _create_range_selector(self, calibration):
        range_string = "Collect References in: " + self._index_range_as_str(
            calibration.frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_mapping_method_selector(self, calibration):
        gazers = user_selectable_gazer_classes()
        gazers_map = gazer_labels_by_class_names(gazers)

        return ui.Selector(
            "gazer_class_name",
            calibration,
            label="Gaze Mapping",
            labels=list(gazers_map.values()),
            selection=list(gazers_map.keys()),
        )

    def _create_min_confidence_slider(self, calibration):
        return ui.Slider(
            "minimum_confidence",
            calibration,
            step=0.01,
            min=0.0,
            max=1.0,
            label="Minimum Pupil Confidence",
        )

    def _create_status_display(self, calibration):
        return ui.Text_Input("status", calibration, label="Status", setter=lambda _: _)

    def _create_calculate_button(self, calibration):
        return ui.Button(
            label="Recalculate" if calibration.params else "Calculate",
            function=self._on_click_calculate,
        )

    def _render_ui_calibration_from_other_recording(self, calibration, menu):
        menu.append(
            ui.Info_Text(
                self._info_text_for_calibration_from_other_recording(calibration)
            )
        )

    def _info_text_for_calibration_from_other_recording(self, calibration):
        gazers = registered_gazer_classes()
        gazer_class_name = calibration.gazer_class_name
        gazer_label = gazer_labels_by_class_names(gazers)[gazer_class_name]

        if calibration.params:
            return (
                f"This {gazer_label} calibration was copied from another recording. "
                "It is ready to be used in gaze mappers."
            )
        else:
            return (
                "This calibration was copied from another recording, but you "
                "cannot use it here, because it is not calculated yet. Please go "
                "back to the original recording, calculate the calibration, "
                "and copy it again."
            )

    def _render_ui_online_calibration(self, calibration, menu):
        menu.append(ui.Info_Text(self._info_text_for_online_calibration(calibration)))

    def _info_text_for_online_calibration(self, calibration):
        gazers = registered_gazer_classes()
        gazer_class_name = calibration.gazer_class_name
        gazer_label = gazer_labels_by_class_names(gazers)[gazer_class_name]

        return (
            f"This {gazer_label} calibration was created before or during the "
            "recording. It is ready to be used in gaze mappers."
        )

    def _on_click_duplicate_button(self):
        if not self._calibration_controller.is_from_same_recording(self.current_item):
            logger.error("Cannot duplicate calibrations from other recordings!")
            return

        if not self.current_item.is_offline_calibration:
            logger.error("Cannot duplicate pre-recorded calibrations!")
            return

        super()._on_click_duplicate_button()

    def _on_click_delete(self):
        if self.current_item is None:
            return

        if not self._calibration_controller.is_from_same_recording(self.current_item):
            logger.error("Cannot delete calibrations from other recordings!")
            return

        if not self.current_item.is_offline_calibration:
            logger.error("Cannot delete pre-recorded calibrations!")
            return

        super()._on_click_delete()

    def _on_name_change(self, new_name):
        self._calibration_storage.rename(self.current_item, new_name)
        # we need to render the menu again because otherwise the name in the selector
        # is not refreshed
        self.render()

    def _on_set_index_range_from_trim_marks(self):
        self._calibration_controller.set_calibration_range_from_current_trim_marks(
            self.current_item
        )
        self.render()

    def _on_click_calculate(self):
        self._calibration_controller.calculate(self.current_item)

    def _on_calibration_computed(self, calibration):
        if calibration == self.current_item:
            # mostly to change button "calculate" -> "recalculate"
            self.render()

    def _on_calculation_could_not_be_started(self):
        self.render()
