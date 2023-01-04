"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import os

import numpy as np
from file_methods import Persistent_Dict
from observable import Observable
from plugin import Plugin

from .storage import ScanPathStorage
from .tasks import ScanPathBackgroundTask, ScanPathPreprocessingTask

logger = logging.getLogger(__name__)


class ScanPathController(Observable):
    """
    Enables previous gaze history to be visible for the timeframe specified by the user.
    """

    min_timeframe = 0.0
    max_timeframe = 3.0
    timeframe_step = 0.05

    def __init__(self, g_pool, **kwargs):
        self.g_pool = g_pool

        self._params = ScanPathParams(**kwargs)
        assert self.min_timeframe <= self.timeframe <= self.max_timeframe, (
            f"min_timeframe={self.min_timeframe}, "
            f"timeframe={self.timeframe}, "
            f"max_timeframe={self.max_timeframe}, "
        )

        self._status_str = ""

        self._preproc = ScanPathPreprocessingTask(g_pool)
        self._preproc.add_observer("on_started", self._on_preproc_started)
        self._preproc.add_observer("on_updated", self._on_preproc_updated)
        self._preproc.add_observer("on_failed", self._on_preproc_failed)
        self._preproc.add_observer("on_canceled", self._on_preproc_canceled)
        self._preproc.add_observer("on_completed", self._on_preproc_completed)

        self._bg_task = ScanPathBackgroundTask(g_pool)
        self._bg_task.add_observer("on_started", self._on_bg_task_started)
        self._bg_task.add_observer("on_updated", self._on_bg_task_updated)
        self._bg_task.add_observer("on_failed", self._on_bg_task_failed)
        self._preproc.add_observer("on_canceled", self._on_bg_task_canceled)
        self._bg_task.add_observer("on_completed", self._on_bg_task_completed)

        self._gaze_data_store = ScanPathStorage(g_pool.rec_dir)
        self._gaze_data_store.load_from_disk()

    def get_init_dict(self):
        return self._params.copy()

    @property
    def timeframe(self) -> float:
        return self._params["timeframe"]

    @timeframe.setter
    def timeframe(self, value: float):
        # It is possible that pyglui.ui.Slider sets this value to a float slighty larger
        # than self.max_timeframe which will trigger the assertion on restoring session
        # settings.
        clipped = max(self.min_timeframe, min(value, self.max_timeframe))
        self._params["timeframe"] = clipped

    @property
    def is_active(self) -> bool:
        return self._preproc.is_active or self._bg_task.is_active

    @property
    def progress(self) -> float:
        if self.is_active:
            ratio = 0.85
            return (
                1.0 - ratio
            ) * self._preproc.progress + ratio * self._bg_task.progress
        else:
            return 0.0  # idle

    @property
    def status_string(self) -> str:
        return self._status_str

    def process(self):
        self._preproc.process()
        self._bg_task.process()

    def scan_path_gaze_for_frame(self, frame):
        if self.timeframe == 0.0:
            return None

        if not self._gaze_data_store.is_valid or not self._gaze_data_store.is_complete:
            if not self.is_active and self.g_pool.app == "player":
                self._trigger_immediate_scan_path_calculation()
            return None

        timestamp_cutoff = frame.timestamp - self.timeframe

        gaze_data = self._gaze_data_store.gaze_data
        gaze_data = gaze_data[gaze_data.frame_index == frame.index]
        gaze_data = gaze_data[gaze_data.timestamp > timestamp_cutoff]

        return gaze_data

    def cleanup(self):
        self._preproc.cleanup()
        self._bg_task.cleanup()
        self._params.cleanup()

    def on_gaze_data_changed(self):
        self._preproc.cancel()
        self._bg_task.cancel()
        self._gaze_data_store.mark_invalid()

    def on_update_ui(self):
        pass

    # Private - helpers

    def _trigger_immediate_scan_path_calculation(self):
        # Cancel old tasks
        self._preproc.cancel()
        self._bg_task.cancel()
        # Start new tasks
        self._preproc.start()

    # Private - preprocessing callbacks

    def _on_preproc_started(self):
        logger.debug("ScanPathController._on_preproc_started")
        self._status_str = "Preprocessing started..."
        self.on_update_ui()

    def _on_preproc_updated(self, gaze_datum):
        self._status_str = f"Preprocessing {int(self._preproc.progress * 100)}%..."
        self.on_update_ui()

    def _on_preproc_failed(self, error):
        logger.debug("ScanPathController._on_preproc_failed")
        logger.error(f"Scan path preprocessing failed: {error}")
        self._status_str = "Preprocessing failed"
        self.on_update_ui()

    def _on_preproc_canceled(self):
        logger.debug("ScanPathController._on_preproc_canceled")
        self._status_str = "Preprocessing canceled"
        self.on_update_ui()

    def _on_preproc_completed(self, gaze_data):
        logger.debug("ScanPathController._on_preproc_completed")
        self._status_str = "Preprocessing completed"
        # Start the background task with max_timeframe
        # The current timeframe will be used only for visualization
        self._bg_task.start(self.max_timeframe, gaze_data)
        self.on_update_ui()

    # Private - calculation callbacks

    def _on_bg_task_started(self):
        logger.debug("ScanPathController._on_bg_task_started")
        self._status_str = "Calculation started..."
        self.on_update_ui()

    def _on_bg_task_updated(self, update_data):
        self._status_str = f"Calculation {int(self._bg_task.progress * 100)}%..."
        # TODO: Save intermediary data
        self.on_update_ui()

    def _on_bg_task_failed(self, error):
        logger.debug("ScanPathController._on_bg_task_failed")
        logger.error(f"Scan path calculation failed: {error}")
        self._status_str = "Calculation failed"
        self.on_update_ui()

    def _on_bg_task_canceled(self):
        logger.debug("ScanPathController._on_bg_task_canceled")
        self._status_str = "Calculation canceled"
        self.on_update_ui()

    def _on_bg_task_completed(self, complete_data):
        logger.debug("ScanPathController._on_bg_task_completed")
        self._gaze_data_store.gaze_data = complete_data
        self._gaze_data_store.mark_complete()
        self._status_str = "Calculation completed"
        self.on_update_ui()


class ScanPathParams(dict):

    version = 1

    default_params = {"timeframe": ScanPathController.min_timeframe}

    def __init__(self, **kwargs):
        super().__init__(**self.default_params)
        self.update(**kwargs)

    def cleanup(self):
        pass
