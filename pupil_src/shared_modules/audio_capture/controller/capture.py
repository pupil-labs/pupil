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

from pupil_audio.nonblocking import PyAudio2PyAVCapture

from observable import Observable


logger = logging.getLogger(__name__)


class AudioCaptureController(Observable):
    def __init__(self):
        self.source_name = None
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

        self._capture = PyAudio2PyAVCapture(
            in_name=self.source_name,
            out_path=self._out_path,
        )
        self._capture.start()
        self._status_string = self._status_str_recording_started()

    def stop_recording(self):
        if not self.is_recording:
            logger.debug("AudioCaptureController.stop_recording called on an already idle instance")
            return
        self._capture.stop()
        self._capture = None
        self._status_string = self._status_str_recording_finished()

    def cleanup(self):
        self.stop_recording()

    # Private

    def _status_str_idle(self) -> str:
        return ""

    def _status_str_recording_started(self) -> str:
        return f"Recording audio from {self.source_name}"

    def _status_str_recording_finished(self) -> str:
        return f"Saved audio to {self._out_path}"
