"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from pyglui import ui

logger = logging.getLogger(__name__)


class OfflineOptimizationMenu:
    menu_label = "Markers 3D Model"

    def __init__(
        self,
        optimization_controller,
        general_settings,
        optimization_storage,
        index_range_as_str,
    ):
        self._optimization_controller = optimization_controller
        self._general_settings = general_settings
        self._optimization_storage = optimization_storage
        self._index_range_as_str = index_range_as_str

        self.menu = ui.Growing_Menu(self.menu_label)
        self.menu.collapsed = False

        optimization_controller.add_observer(
            "on_optimization_completed", self._on_optimization_completed
        )

    def render(self):
        self.menu.elements.clear()
        self._render_ui()

    def _render_ui(self):
        if self._optimization_storage.is_from_same_recording:
            self.menu.elements.extend(
                self._render_ui_markers_3d_model_from_same_recording()
            )
        else:
            self.menu.elements.extend(
                self._render_ui_markers_3d_model_from_another_recording()
            )

    def _render_ui_markers_3d_model_from_same_recording(self):
        menu = [
            self._create_name_input(),
            self._create_range_selector(),
            self._create_optimize_camera_intrinsics_switch(),
            self._create_origin_marker_id_display_from_same_recording(),
            self._create_calculate_button(),
            self._create_status_display(),
        ]
        return menu

    def _render_ui_markers_3d_model_from_another_recording(self):
        menu = [
            self._create_info_text_for_markers_3d_model_from_another_recording(),
            self._create_origin_marker_id_display_from_another_recording(),
        ]
        return menu

    def _create_info_text_for_markers_3d_model_from_another_recording(self):
        if self._optimization_storage.calculated:
            text = (
                "This markers 3d model '{}' was copied from another recording. "
                "It is ready to be used for camera localization.".format(
                    self._optimization_storage.name
                )
            )
        else:
            text = (
                "This markers 3d model '{}' was copied from another recording, "
                "but it cannot be used here, since it was not successfully calculated. "
                "Please go back to the original recording, calculate and copy it here "
                "again.".format(self._optimization_storage.name)
            )
        return ui.Info_Text(text)

    def _create_name_input(self):
        return ui.Text_Input(
            "name",
            self._optimization_storage,
            label="Name",
            setter=self._on_name_change,
        )

    def _create_range_selector(self):
        range_string = "Collect markers in: " + self._index_range_as_str(
            self._general_settings.optimization_frame_index_range
        )
        return ui.Button(
            outer_label=range_string,
            label="Set from trim marks",
            function=self._on_set_index_range_from_trim_marks,
        )

    def _create_optimize_camera_intrinsics_switch(self):
        return ui.Switch(
            "optimize_camera_intrinsics",
            self._general_settings,
            label="Optimize camera intrinsics",
        )

    def _create_calculate_button(self):
        return ui.Button(
            label="Recalculate"
            if self._optimization_storage.calculated
            else "Calculate",
            function=self._on_calculate_button_clicked,
        )

    def _create_status_display(self):
        return ui.Text_Input(
            "status", self._optimization_controller, label="Status", setter=lambda _: _
        )

    def _create_origin_marker_id_display_from_same_recording(self):
        return ui.Text_Input(
            "user_defined_origin_marker_id",
            self._general_settings,
            label="Define the origin marker id",
            getter=self._on_get_origin_marker_id,
            setter=self._on_set_origin_marker_id,
        )

    def _create_origin_marker_id_display_from_another_recording(self):
        return ui.Text_Input(
            "user_defined_origin_marker_id",
            self._general_settings,
            label="Origin marker id",
            getter=self._on_get_origin_marker_id,
            setter=lambda _: _,
        )

    def _on_name_change(self, new_name):
        self._optimization_storage.rename(new_name)
        self.render()

    def _on_set_index_range_from_trim_marks(self):
        self._optimization_controller.set_range_from_current_trim_marks()
        self.render()

    def _on_calculate_button_clicked(self):
        self._optimization_controller.calculate()
        self.render()

    def _on_get_origin_marker_id(self):
        if (
            self._optimization_storage.is_from_same_recording
            and self._general_settings.user_defined_origin_marker_id is not None
        ):
            origin_marker_id = self._general_settings.user_defined_origin_marker_id
        elif self._optimization_storage.calculated:
            origin_marker_id = self._optimization_storage.origin_marker_id
        else:
            origin_marker_id = None
        return str(origin_marker_id)

    def _on_set_origin_marker_id(self, new_id):
        try:
            new_id = int(new_id)
        except ValueError:
            logger.info(f"'{new_id}' is not a valid input")
            return

        if self._general_settings.user_defined_origin_marker_id != new_id:
            self._general_settings.user_defined_origin_marker_id = new_id
            logger.info(
                "The marker with id {} will be defined as the origin of the"
                "coordinate system during the next calculation of the markers 3d "
                "model '{}'.".format(new_id, self._optimization_storage.name)
            )
            self._optimization_controller.calculate()

    def _on_optimization_completed(self):
        self.render()
