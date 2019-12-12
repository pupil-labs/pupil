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
from audio_capture.model.caputre import PyAudio2PyAVObservableMultipartCapture


logger = logging.getLogger(__name__)


class AudioCaptureController(Observable):
    def __init__(self, device_monitor=None):
        self.source_name = None
        self.device_monitor = device_monitor
        # Private
        self._status_string = self._status_str_idle()
        self._capture = None
        self._out_path = None

    @property
    def is_recording(self) -> bool:
        return self._capture is not None

    @property
    def status_string(self) -> str:
        return self._status_string

    def start_recording(self, out_path: str):
        if self.source_name is None:
            logger.debug("No audio source selected. Skipping audio capture.")
            return

        if self.is_recording:
            logger.debug("AudioCaptureController.start_recording called on an already recording instance")
            return

        self._out_path = out_path

        self._capture = PyAudio2PyAVObservableMultipartCapture(
            in_name=self.source_name,
            out_path=self._out_path,
            device_monitor=self.device_monitor,
        )

        self._capture.add_observer("on_input_device_connected", self.on_input_device_connected)
        self._capture.add_observer("on_input_device_disconnected", self.on_input_device_disconnected)

        self._capture.start()
        self._status_string = self._status_str_recording_started()
        self.on_status_update()

    def stop_recording(self):
        if not self.is_recording:
            logger.debug("AudioCaptureController.stop_recording called on an already idle instance")
            return
        self._capture.stop()
        self._capture = None
        self._status_string = self._status_str_idle()
        self.on_status_update()

    def cleanup(self):
        self.stop_recording()

    def on_status_update(self):
        pass

    # Private

    def _status_str_idle(self) -> str:
        return ""

    def _status_str_recording_started(self) -> str:
        return f"Recording audio from {self.source_name}"

    def _status_str_recording_paused(self) -> str:
        return f"Waiting for audio device {self.source_name}"

    def on_input_device_connected(self, device_info):
        self._status_string = self._status_str_recording_started()
        self.on_status_update()

    def on_input_device_disconnected(self):
        self._status_string = self._status_str_recording_paused()
        self.on_status_update()
