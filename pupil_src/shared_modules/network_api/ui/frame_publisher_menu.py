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

from network_api.controller import FramePublisherController
from network_api.model import FrameFormat
from observable import Observable
from pyglui import ui

logger = logging.getLogger(__name__)


class FramePublisherMenu(Observable):
    menu_label = "Frame Publisher"

    def __init__(self, frame_publisher_controller: FramePublisherController):
        self.__sub_menu = None
        self.__frame_publisher_controller = frame_publisher_controller

    def append_to_menu(self, menu):
        if self.__sub_menu is None:
            self.__sub_menu = ui.Growing_Menu(self.menu_label)
        menu.append(self.__sub_menu)
        self._update_menu()

    ### PRIVATE

    def _update_menu(self):
        self.__remove_menu_items()
        self.__insert_menu_items()

    def __remove_menu_items(self):
        if self.__sub_menu is None:
            return
        del self.__sub_menu.elements[:]

    def __insert_menu_items(self):
        if self.__sub_menu is None:
            return

        format_values = FrameFormat.available_formats()
        format_labels = [f.label for f in format_values]

        ui_info_text = ui.Info_Text(
            'Publishes frame data in different formats under the topics "frame.world", '
            '"frame.eye.0", and "frame.eye.1".'
        )
        ui_selector_format = ui.Selector(
            "frame_format",
            self.__frame_publisher_controller,
            label="Format",
            selection=format_values,
            labels=format_labels,
        )

        self.__sub_menu.append(ui_info_text)
        self.__sub_menu.append(ui_selector_format)
