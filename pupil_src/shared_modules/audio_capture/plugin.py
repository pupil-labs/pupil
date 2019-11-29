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

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.parse_pretty_class_name()
        self.menu.append(ui.Info_Text(self.__doc__))

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        pass

    def on_notify(self, notification):
        pass

    # Private
