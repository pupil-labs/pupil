"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
import logging
import typing as T

import stdlib_utils
from plugin import Plugin
from pyglui import ui

from .controller.source import AudioSourceController


class AudioCapturePlugin(Plugin):
    """Creates events for audio input.
    """

    icon_chr = chr(0xE029)
    icon_font = "pupil_icons"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Audio Capture"

    def __init__(self, g_pool, audio_src=None):
        super().__init__(g_pool)
        self.source_controller = AudioSourceController(audio_src)
        self.source_controller.add_observer("on_selected", self._on_source_selected)

    def get_init_dict(self):
        return {"audio_src": self.source_controller.current_source}

    ui_source_selector = stdlib_utils.lazy_property(
        lambda self: ui.Selector(
            "source_selector",
            None,
            label="Audio Source",
            getter=self.source_controller.ui_selector_current_getter,
            setter=self.source_controller.ui_selector_current_setter,
            selection_getter=self.source_controller.ui_selector_enumerate_options,
        )
    )

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.parse_pretty_class_name()
        self.menu.append(ui.Info_Text(self.__doc__))
        self.menu.append(self.ui_source_selector)

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        self.ui_source_selector.read_only = self._is_audio_busy

    def on_notify(self, notification):
        pass

    # Private

    def _on_source_selected(self, name: T.Optional[str]):
        print(f"===> SELECTED NEW AUDIO SOURCE: {name}") #FIXME