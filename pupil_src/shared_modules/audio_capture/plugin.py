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
from .controller.capture import AudioCaptureController
from .controller.mic_check import AudioMicCheckController


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

        self.capture_controller = AudioCaptureController()

        self.mic_check_controller = AudioMicCheckController()

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

    ui_status_text = stdlib_utils.lazy_property(
        lambda self: ui.Info_Text(
            self.capture_controller.status_string,
        )
    )

    ui_mic_check_button = stdlib_utils.lazy_property(
        lambda self: ui.Button(
            label="Mic Check",
            function=self.mic_check_controller.perform_check,
        )
    )

    ui_mic_check_status_text = stdlib_utils.lazy_property(
        lambda self: ui.Info_Text(
            self.mic_check_controller.status_string,
        )
    )

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.parse_pretty_class_name()
        self.menu.append(ui.Info_Text(self.__doc__))
        # Input selection section
        self.menu.append(ui.Separator())
        self.menu.append(self.ui_source_selector)
        self.menu.append(self.ui_status_text)
        # Mic check section
        self.menu.append(ui.Separator())
        self.menu.append(self.ui_mic_check_button)
        self.menu.append(self.ui_mic_check_status_text)

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        self.ui_source_selector.read_only = self._is_audio_busy
        self.ui_status_text.text = self.capture_controller.status_string
        self.ui_mic_check_button.read_only = not self._is_audio_busy
        self.ui_mic_check_status_text.text = self.mic_check_controller.status_string

    def on_notify(self, notification):
        if notification["subject"] == "recording.started":
            self._on_recording_started(rec_dir=notification["rec_path"])
        elif notification["subject"] == "recording.stopped":
            self._on_recording_stopped()

    def cleanup(self):
        self.source_controller.cleanup()
        self.capture_controller.cleanup()
        self.mic_check_controller.cleanup()

    # Private

    @property
    def _is_audio_busy(self) -> bool:
        return self.capture_controller.is_recording or self.mic_check_controller.is_checking

    def _on_source_selected(self, name: T.Optional[str]):
        self.capture_controller.source_name = name
        self.mic_check_controller.source_name = name

    def _on_recording_started(self, rec_dir):
        out_path = os.path.join(rec_dir, "audio.mp4")
        self.capture_controller.start_recording(out_path)

    def _on_recording_stopped(self):
        self.capture_controller.stop_recording()
