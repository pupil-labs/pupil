"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import time
import logging
import threading
import typing as T

from observable import Observable

from pupil_audio.utils.pyaudio import DeviceInfo
from pupil_audio.utils.pyaudio import get_all_inputs as pyaudio_get_all_inputs
from pupil_audio.utils.pyaudio import create_session as pyaudio_create_session
from pupil_audio.utils.pyaudio import destroy_session as pyaudio_destroy_session


_AUDIO_SOURCE_NONE_NAME = "No Audio"


logger = logging.getLogger(__name__)


class AudioSourceController(Observable):
    def __init__(self, audio_src=None):
        self._current_source = None
        self._discovered_sources = []

        self._monitor = AudioSourceMonitor()
        self._monitor.add_observer("on_device_connected", self._on_device_connected)
        self._monitor.add_observer("on_device_updated", self._on_device_updated)
        self._monitor.add_observer("on_device_disconnected", self._on_device_disconnected)
        self._monitor.start_monitoring()

        self._set_source_if_valid(audio_src)

    # Public

    @property
    def current_source(self) -> T.Optional[str]:
        return self._current_source

    def cleanup(self):
        self._monitor.cleanup()

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
        names = self._monitor.devices_by_name.keys()
        assert _AUDIO_SOURCE_NONE_NAME not in names

        names = list(names)
        names.insert(0, _AUDIO_SOURCE_NONE_NAME)

        self._discovered_sources = names

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


class AudioSourceMonitor(Observable):
    def __init__(self):
        self._session = pyaudio_create_session()
        self._devices_by_name = self._get_input_devices_by_name()
        self._should_run = threading.Event()
        self._monitor_thread = None

    @property
    def is_running(self):
        return self._should_run.is_set()

    @property
    def devices_by_name(self) -> T.Mapping[str, DeviceInfo]:
        return self._devices_by_name

    @devices_by_name.setter
    def devices_by_name(self, new_devices_by_name: DeviceInfo):
        old_devices_by_name = self._devices_by_name

        old_names = set(old_devices_by_name.keys())
        new_names = set(new_devices_by_name.keys())

        connected_names = new_names.difference(old_names)
        existing_names = new_names.intersection(old_names)
        disconnected_names = old_names.difference(new_names)

        for name in connected_names:
            device_info = new_devices_by_name[name]
            old_devices_by_name[name] = device_info
            self.on_device_connected(device_info)

        for name in existing_names:
            device_info = new_devices_by_name[name]
            if device_info != old_devices_by_name[name]:
                old_devices_by_name[name] = device_info
                self.on_device_updated(device_info)

        for name in disconnected_names:
            device_info = old_devices_by_name[name]
            del old_devices_by_name[name]
            self.on_device_disconnected(device_info)

        self._devices_by_name = old_devices_by_name

    def start_monitoring(self):
        if self.is_running:
            return
        self._should_run.set()
        self._monitor_thread = threading.Thread(
            name=type(self).__class__.__name__,
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._should_run.clear()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self._monitor_thread = None

    def cleanup(self):
        self.stop_monitoring()
        if self._session is not None:
            pyaudio_destroy_session(self._session)
            self._session = None

    def on_device_connected(self, device_info):
        pass

    def on_device_updated(self, device_info):
        pass

    def on_device_disconnected(self, device_info):
        pass

    def _get_input_devices_by_name(self):
        return pyaudio_get_all_inputs(unowned_session=self._session)

    def _monitor_loop(self, sleep_time=3):
        while self.is_running:
            try:
                x = self._get_input_devices_by_name()
                self.devices_by_name = x
                time.sleep(sleep_time)
            except Exception as err:
                logger.error(err)
