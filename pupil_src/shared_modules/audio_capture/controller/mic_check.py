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
from pathlib import Path

from pupil_audio.nonblocking import PyAudio2PyAVCapture

from stdlib_utils import _create_temporary_unique_file_path
from observable import Observable


logger = logging.getLogger(__name__)


class AudioMicCheckController(Observable):
    def __init__(self):
        # Public
        self.source_name = None
        self.mic_check_duration_sec = 3
        # Private
        self._status_string = self._status_str_idle()
        self._checking_thread = None

    # Public

    @property
    def is_checking(self) -> bool:
        return self._checking_thread is not None

    @property
    def can_perform_check(self) -> bool:
        return self.source_name is not None

    @property
    def status_string(self) -> str:
        return self._status_string

    def perform_check(self):
        if self.source_name is None:
            logger.debug("No audio source selected. Skipping audio capture.")
            return
        if self.is_checking:
            logger.debug("AudioMicCheckController.perform_check called on an already busy instance")
            return
        self._checking_thread = threading.Thread(
            name=type(self).__class__.__name__,
            target=self._mic_check_fn,
            args=(self.source_name, self.mic_check_duration_sec),
        )
        self._checking_thread.start()

    # Callbacks

    def on_check_started(self):
        pass

    def on_check_finished(self):
        pass

    # Private

    def _status_str_idle(self) -> str:
        return ""

    def _status_str_checking(self) -> str:
        return "Checking mic..."

    def _status_str_success(self) -> str:
        return "Mic check successfull"

    def _status_str_failure(self, reason) -> str:
        return f"Mic check failed: {reason}"

    def _mic_check_fn(self, in_name, duration):

        out_path = _create_temporary_unique_file_path(ext=".mp4")
        start_time = time.monotonic()
        sleep_step = 0.1
        capture = None

        def _report_failure(reason):
            self._status_string = self._status_str_failure(reason=reason)
            self.on_check_finished()

        def _report_success():
            self._status_string = self._status_str_success()
            self.on_check_finished()

        self.on_check_started()

        try:
            capture = PyAudio2PyAVCapture(
                in_name=in_name,
                out_path=str(out_path),
            )
            capture.start()
            while time.monotonic() - start_time < duration:
                time.sleep(sleep_step)
        except Exception as err:
            return _report_failure(err)
        finally:
            if capture:
                capture.stop()
            self._checking_thread = None

        is_output_valid, failure_reason = self._validate_out_file(out_path)

        if not is_output_valid:
            return _report_failure(failure_reason)

        out_path.unlink()
        return _report_success()

    @staticmethod
    def _validate_out_file(out_path: Path) -> T.Tuple[bool, str]:
        if not out_path.is_file():
            return (False, "Recorded file was not saved")

        if out_path.stat().st_size == 0:
            return (False, "Recorded file is empty")

        return (True, "")
