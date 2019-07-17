"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
import weakref
import eye_movement.model as model
from pyglui import ui as gl_ui


class Menu_Content:
    def __init__(self, plugin, label_text: str, show_segmentation: bool):
        self.plugin = weakref.ref(plugin)
        self.label_text = label_text
        self.detection_status = ""
        self.show_segmentation = show_segmentation

        self._show_segmentation_switch = None
        self._current_segment_details = None
        self._detection_status_input = None
        self._error_text = None

    #

    def _info_paragraphs(self) -> t.List[str]:
        plugin = self.plugin()
        if not plugin:
            return []

        def clean_paragraph(p: str) -> str:
            return p.replace("\n", " ").replace("  ", "").strip()

        text = str(plugin.__doc__)
        paragraphs = [clean_paragraph(p) for p in text.split("\n\n")]
        paragraphs.append("Press the export button or type 'e' to start the export.")
        return paragraphs

    def add_to_menu(self, menu):

        self._detection_status_input = gl_ui.Text_Input(
            "detection_status", self, label="Detection progress:", setter=lambda _: None
        )

        self._show_segmentation_switch = gl_ui.Switch(
            "show_segmentation", self, label="Show segmentation"
        )

        self._error_text = gl_ui.Info_Text("")
        self._current_segment_details = gl_ui.Info_Text("")

        menu.label = self.label_text

        for paragraph in self._info_paragraphs():
            paragraph_ui = gl_ui.Info_Text(paragraph)
            menu.append(paragraph_ui)

        menu.append(self._show_segmentation_switch)
        menu.append(self._detection_status_input)
        menu.append(self._error_text)
        menu.append(self._current_segment_details)

    def update_status(self, new_status: str):
        self.detection_status = new_status

    def update_progress(self, new_progress: float):
        plugin = self.plugin()
        if plugin and plugin.menu_icon:
            plugin.menu_icon.indicator_stop = new_progress

    def update_error_text(self, error_message: str):
        error_message = f"Error: {error_message}" if error_message else ""
        self._error_text.text = error_message

    def update_detail_text(
        self,
        current_index: int,
        total_segment_count: int,
        current_segment: t.Optional[model.Classified_Segment],
        prev_segment: t.Optional[model.Classified_Segment],
        next_segment: t.Optional[model.Classified_Segment],
    ):

        if not self._current_segment_details:
            return

        if (
            (current_index is None)
            or (total_segment_count < 1)
            or (not current_segment)
        ):
            self._current_segment_details.text = ""
            return

        text = ""
        ident = "    "

        text += "Current segment, {} of {}\n".format(
            current_index + 1, total_segment_count
        )
        text += ident + "ID: {}\n".format(current_segment.id)
        text += ident + "Classification: {}\n".format(
            current_segment.segment_class.value
        )
        text += ident + "Confidence: {:.2f}\n".format(current_segment.confidence)
        text += ident + "Duration: {:.2f} milliseconds\n".format(
            current_segment.duration
        )
        text += ident + "Frame range: {}-{}\n".format(
            current_segment.start_frame_index, current_segment.end_frame_index
        )
        text += ident + "2d gaze pos: x={:.3f}, y={:.3f}\n".format(
            *current_segment.norm_pos
        )
        if current_segment.gaze_point_3d:
            text += ident + "3d gaze pos: x={:.3f}, y={:.3f}, z={:.3f}\n".format(
                *current_segment.gaze_point_3d
            )
        else:
            text += ident + "3d gaze pos: N/A\n"

        if prev_segment:
            text += ident + "Time since prev. segment: {:.2f} seconds\n".format(
                prev_segment.duration / 1000
            )
        else:
            text += ident + "Time since prev. segment: N/A\n"

        if next_segment:
            text += ident + "Time to next segment: {:.2f} seconds\n".format(
                current_segment.duration / 1000
            )
        else:
            text += ident + "Time to next segment: N/A\n"

        self._current_segment_details.text = text
