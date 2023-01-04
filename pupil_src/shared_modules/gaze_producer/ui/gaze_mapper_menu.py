"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from gaze_producer import ui as plugin_ui
from pyglui import ui


class GazeMapperMenu(plugin_ui.StorageEditMenu):
    menu_label = "Gaze Mappers"
    selector_label = "Edit Gaze Mapper:"
    new_button_label = "New Gaze Mapper"
    duplicate_button_label = "Duplicate Current Gaze Mapper"

    def __init__(
        self,
        gaze_mapper_controller,
        gaze_mapper_storage,
        calibration_storage,
        index_range_as_str,
    ):
        super().__init__(gaze_mapper_storage)
        self._gaze_mapper_controller = gaze_mapper_controller
        self._gaze_mapper_storage = gaze_mapper_storage
        self._calibration_storage = calibration_storage
        self._index_range_as_str = index_range_as_str

        self.menu.collapsed = True

        calibration_storage.add_observer("add", self._on_calibration_storage_changed)
        calibration_storage.add_observer("rename", self._on_calibration_storage_changed)
        calibration_storage.add_observer("delete", self._on_calibration_storage_changed)

        gaze_mapper_controller.add_observer(
            "on_gaze_mapping_calculated", self._on_gaze_mapping_calculated
        )
        gaze_mapper_controller.add_observer(
            "on_calculation_could_not_be_started",
            self._on_calculation_could_not_be_started,
        )

    def _item_label(self, gaze_mapper):
        return gaze_mapper.name

    def _new_item(self):
        return self._gaze_mapper_storage.create_default_gaze_mapper()

    def _duplicate_item(self, gaze_mapper):
        return self._gaze_mapper_storage.duplicate_gaze_mapper(gaze_mapper)

    def _render_custom_ui(self, gaze_mapper, menu):
        menu.extend(
            [
                self._create_name_input(gaze_mapper),
                self._create_status_text(gaze_mapper),
                self._create_calculate_button(gaze_mapper),
                self._create_calibration_selector(gaze_mapper),
                self._create_mapping_range_selector(gaze_mapper),
                self._create_manual_correction_submenu(gaze_mapper),
                self._create_validation_submenu(gaze_mapper),
                self._create_activate_gaze_switch(gaze_mapper),
            ]
        )

    def _create_name_input(self, gaze_mapper):
        return ui.Text_Input(
            "name", gaze_mapper, label="Name", setter=self._on_name_change
        )

    def _create_status_text(self, gaze_mapper):
        return ui.Text_Input("status", gaze_mapper, label="Status", setter=lambda _: _)

    def _create_calculate_button(self, gaze_mapper):
        return ui.Button(
            label="Calculate" if gaze_mapper.empty() else "Recalculate",
            function=self._on_click_calculate,
        )

    def _create_calibration_selector(self, gaze_mapper):
        labels = [calibration.name for calibration in self._calibration_storage]
        selection = [calibration.unique_id for calibration in self._calibration_storage]

        calibration = self._gaze_mapper_controller.get_valid_calibration_or_none(
            gaze_mapper
        )
        if not calibration:
            labels.append("[Invalid Calibration]")
            selection.append(gaze_mapper.calibration_unique_id)

        def calibration_setter(calibration_unique_id):
            self._gaze_mapper_controller.set_calibration_unique_id(
                gaze_mapper, calibration_unique_id
            )

        return ui.Selector(
            "calibration_unique_id",
            gaze_mapper,
            label="Calibration",
            selection=selection,
            labels=labels,
            setter=calibration_setter,
        )

    def _create_mapping_range_selector(self, gaze_mapper):
        range_string = "Map gaze in: " + self._index_range_as_str(
            gaze_mapper.mapping_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_mapping_range_from_trim_marks,
        )

    def _create_manual_correction_submenu(self, gaze_mapper):
        manual_correction_submenu = ui.Growing_Menu("Manual Correction")
        manual_correction_submenu.extend(
            [
                self._create_manual_correction_slider("x", gaze_mapper),
                self._create_manual_correction_slider("y", gaze_mapper),
            ]
        )
        manual_correction_submenu.collapsed = True
        return manual_correction_submenu

    def _create_manual_correction_slider(self, axis, gaze_mapper):
        return ui.Slider(
            "manual_correction_" + axis,
            gaze_mapper,
            min=-0.5,
            step=0.01,
            max=0.5,
            label="Manual Correction " + axis.upper(),
        )

    def _create_validation_submenu(self, gaze_mapper):
        validation_submenu = ui.Growing_Menu("Validation")
        validation_submenu.extend(
            [
                self._create_validation_range_selector(gaze_mapper),
                self._create_outlier_threshold_selector(gaze_mapper),
                self._create_accuracy_output(gaze_mapper),
                self._create_precision_output(gaze_mapper),
            ]
        )
        validation_submenu.collapsed = True
        return validation_submenu

    def _create_validation_range_selector(self, gaze_mapper):
        range_string = "Validate in: " + self._index_range_as_str(
            gaze_mapper.validation_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set From Trim Marks",
            function=self._on_set_validation_range_from_trim_marks,
        )

    def _create_outlier_threshold_selector(self, gaze_mapper):
        return ui.Text_Input(
            "validation_outlier_threshold_deg",
            gaze_mapper,
            label="Outlier Threshold [degrees]",
            setter=self._on_outlier_threshold_changed,
        )

    def _create_accuracy_output(self, gaze_mapper):
        return ui.Text_Input(
            "accuracy_result", gaze_mapper, label="Accuracy Result", setter=lambda _: _
        )

    def _create_precision_output(self, gaze_mapper):
        return ui.Text_Input(
            "precision_result",
            gaze_mapper,
            label="Precision Result",
            setter=lambda _: _,
        )

    def _create_activate_gaze_switch(self, gaze_mapper):
        return ui.Switch(
            "activate_gaze",
            gaze_mapper,
            label="Activate Gaze",
            setter=self._on_activate_gaze_changed,
        )

    def _on_calibration_storage_changed(self, *args, **kwargs):
        self.render()

    def _on_name_change(self, new_name):
        self._gaze_mapper_storage.rename(self.current_item, new_name)
        # we need to render the menu again because otherwise the name in the selector
        # is not refreshed
        self.render()

    def _on_set_mapping_range_from_trim_marks(self):
        self._gaze_mapper_controller.set_mapping_range_from_current_trim_marks(
            self.current_item
        )
        self.render()

    def _on_set_validation_range_from_trim_marks(self):
        self._gaze_mapper_controller.set_validation_range_from_current_trim_marks(
            self.current_item
        )
        self._gaze_mapper_controller.validate_gaze_mapper(self.current_item)
        self.render()

    def _on_outlier_threshold_changed(self, new_threshold):
        self.current_item.validation_outlier_threshold_deg = new_threshold
        self._gaze_mapper_controller.validate_gaze_mapper(self.current_item)

    def _on_activate_gaze_changed(self, new_value):
        self.current_item.activate_gaze = new_value
        self._gaze_mapper_controller.publish_all_enabled_mappers()

    def _on_click_calculate(self):
        self._gaze_mapper_controller.calculate(self.current_item)

    def _on_gaze_mapping_calculated(self, gaze_mapping):
        if gaze_mapping == self.current_item:
            # mostly to change button "calculate" -> "recalculate"
            self.render()

    def _on_calculation_could_not_be_started(self):
        self.render()
