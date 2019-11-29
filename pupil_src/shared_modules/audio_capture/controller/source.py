"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as T

from observable import Observable

from pupil_audio.utils.pyaudio import get_all_inputs as pyaudio_get_all_inputs


_AUDIO_SOURCE_NONE_NAME = "No Audio"


class AudioSourceController(Observable):
    def __init__(self, audio_src=None):
        self._current_source = None
        self._cached_sources = []
        self._set_source_if_valid(audio_src)

    # Public

    def current_source(self) -> T.Optional[str]:
        return self._current_source

    # Callbacks

    def on_selected(self, name: T.Optional[str]):
        pass

    # UI Selector

    def ui_selector_current_getter(self) -> str:
        return self._current_source or _AUDIO_SOURCE_NONE_NAME

    def ui_selector_current_setter(self, name: str):
        self._set_source_if_valid(name)

    def ui_selector_enumerate_options(self) -> T.List[str]:
        options = self._get_source_names()
        return options, options

    # Private

    def _set_source_if_valid(self, name):
        source_names = self._cached_sources or self._get_source_names()

        if name == _AUDIO_SOURCE_NONE_NAME:
            new_valid_source = None

        elif name in source_names:
            new_valid_source = name

        elif self._current_source not in source_names:
            new_valid_source = None

        if self._current_source != new_valid_source:
            self._current_source = new_valid_source
            self.on_selected(new_valid_source)

    def _get_source_names(self) -> T.List[str]:
        names = pyaudio_get_all_inputs().keys()
        assert _AUDIO_SOURCE_NONE_NAME not in names

        names = list(names)
        names.insert(0, _AUDIO_SOURCE_NONE_NAME)

        self._cached_sources = names
        return names
