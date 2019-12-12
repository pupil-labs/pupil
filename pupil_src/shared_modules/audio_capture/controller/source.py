"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

from observable import Observable
from pupil_capture.model.monitor import PyAudioObservableBackgroundDeviceMonitor


_AUDIO_SOURCE_NONE_NAME = "No Audio"


logger = logging.getLogger(__name__)


class AudioSourceController(Observable):
    def __init__(self, audio_src=None):
        self._current_source = None
        self._discovered_sources = []

        self.device_monitor = PyAudioObservableBackgroundDeviceMonitor()
        self.device_monitor.add_observer("on_device_connected", self._on_device_connected)
        self.device_monitor.add_observer("on_device_updated", self._on_device_updated)
        self.device_monitor.add_observer("on_device_disconnected", self._on_device_disconnected)
        self.device_monitor.start()

        self._set_source_if_valid(audio_src)

    # Public

    @property
    def current_source(self) -> T.Optional[str]:
        return self._current_source

    def cleanup(self):
        self.device_monitor.cleanup()

    # Callbacks

    def on_selected(self, name: T.Optional[str]):
        pass

    # UI Selector

    def ui_selector_current_getter(self) -> str:
        return self._current_source or _AUDIO_SOURCE_NONE_NAME

    def ui_selector_current_setter(self, name: str):
        self._set_source_if_valid(name)

    def ui_selector_enumerate_options(self) -> T.List[str]:
        options = self._discovered_sources
        return options, options

    # Private

    def _on_device_connected(self, device_info):
        self._update_discovered_sources()

    def _on_device_updated(self, device_info):
        self._update_discovered_sources()

    def _on_device_disconnected(self, device_info):
        self._update_discovered_sources()

    def _update_discovered_sources(self):
        devices_by_name = self.device_monitor.devices_by_name

        input_names = [name for name, device_info in devices_by_name.items() if device_info.is_input]
        assert _AUDIO_SOURCE_NONE_NAME not in input_names

        input_names.insert(0, _AUDIO_SOURCE_NONE_NAME)

        self._discovered_sources = input_names

    def _set_source_if_valid(self, name):
        self._update_discovered_sources()

        if name == _AUDIO_SOURCE_NONE_NAME:
            new_valid_source = None

        elif name in self._discovered_sources:
            new_valid_source = name

        elif self._current_source not in self._discovered_sources:
            new_valid_source = None

        if self._current_source != new_valid_source:
            self._current_source = new_valid_source
            self.on_selected(new_valid_source)
