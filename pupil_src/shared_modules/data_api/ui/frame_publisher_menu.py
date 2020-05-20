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

from observable import Observable
from pyglui import ui

from data_api.model import FrameFormat
from data_api.controller import FramePublisherController


logger = logging.getLogger(__name__)


class FramePublisherMenu(Observable):
    menu_label = "Frame Publisher"

    def __init__(self, frame_publisher_controller: FramePublisherController):
        self._frame_publisher_controller = frame_publisher_controller

    def append_to_menu(self, menu):
        format_values = FrameFormat.available_formats()
        format_labels = [f.label for f in format_values]

        ui_info_text = ui.Info_Text(
            'Publishes frame data in different formats under the topic "frame.world".'
        )
        ui_selector_format = ui.Selector(
            "frame_format",
            self._frame_publisher_controller,
            label="Format",
            selection=format_values,
            labels=format_labels,
        )

        sub_menu = ui.Growing_Menu(self.menu_label)
        sub_menu.append(ui_info_text)
        sub_menu.append(ui_selector_format)
        menu.append(sub_menu)
