"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging

import numpy as np

from observable import Observable
from plugin import Plugin

from .scan_path_storage import ScanPathItem, ScanPathStorage
from .scan_path_preprocessing_task import ScanPathPreprocessingTask
from .scan_path_background_task import ScanPathBackgroundTask


logger = logging.getLogger(__name__)


class ScanPathController(Observable):

    min_timeframe = 1.0
    max_timeframe = 5.0

    def __init__(self, g_pool, timeframe=3.0):
        self.g_pool = g_pool

        assert self.min_timeframe <= timeframe <= self.max_timeframe
        self.timeframe = timeframe

        self._status_str = ""
        self._preproc_data = []
        self._computed_storage = ScanPathStorage(self.g_pool.rec_dir, self)

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

        if self._computed_storage.is_completed:
            self._status_str = "Loaded from cache"
        else:
            self._computed_storage.clear()
            self._trigger_delayed_scan_path_calculation()

    def get_init_dict(self):
        return {"timeframe": self.timeframe}

    @property
    def is_active(self) -> bool:
        return self._preproc.is_active or self._bg_task.is_active

    @property
    def progress(self) -> float:
        if self.is_active:
            ratio = 0.9
            return (1.0-ratio) * self._preproc.progress + ratio * self._bg_task.progress
        else:
            return 0.0  #idle

    @property
    def status_string(self) -> str:
        return self._status_str

    def process(self):
        self._preproc.process()
        self._bg_task.process()

    def gaze_data_at_frame_index(self, frame_index):
        if self.is_active:
            return None
        else:
            return self._computed_storage.get(frame_index)

    def cleanup(self):
        pass

    def on_notify(self, notification):
        if notification["subject"] == self._recalculate_scan_path_notification_subject:
            self._trigger_immediate_scan_path_calculation()
        elif notification["subject"] == "gaze_positions_changed":
            self._trigger_immediate_scan_path_calculation()

    def on_update_ui(self):
        pass

    # Private - helpers

    _recalculate_scan_path_notification_subject = "scan_path.should_recalculate"

    def _trigger_delayed_scan_path_calculation(self, delay=1.0):
        Plugin.notify_all(self, {"subject": self._recalculate_scan_path_notification_subject, "delay": delay})

    def _trigger_immediate_scan_path_calculation(self):
        # Cancel old tasks
        self._preproc.cancel()
        self._bg_task.cancel()
        # Clear old data
        self._clear_data()
        # Start new tasks
        self._preproc.start()

    def _clear_data(self):
        self._preproc_data = []
        self._computed_storage.clear()

    # Private - preprocessing callbacks

    def _on_preproc_started(self):
        logger.debug("ScanPathController._on_preproc_started")
        self._status_str = "Preprocessing started..."
        self._clear_data()
        self.on_update_ui()

    def _on_preproc_updated(self, update_data):
        logger.debug("ScanPathController._on_preproc_updated")
        assert len(self._preproc_data) == update_data.frame.index, "Frames must be processed consecutively"
        self._preproc_data.append(update_data.gaze_data)
        self._status_str = f"Preprocessing {int(self._preproc.progress * 100)}%..."
        self.on_update_ui()

    def _on_preproc_failed(self, error):
        logger.debug("ScanPathController._on_preproc_failed")
        logger.error(f"Scan path preprocessing failed: {error}")
        self._status_str = "Preprocessing failed"
        self._clear_data()
        self.on_update_ui()

    def _on_preproc_canceled(self):
        logger.debug("ScanPathController._on_preproc_canceled")
        self._status_str = "Preprocessing canceled"
        self._clear_data()
        self.on_update_ui()

    def _on_preproc_completed(self):
        logger.debug("ScanPathController._on_preproc_completed")
        self._status_str = "Preprocessing completed"
        gaze_data = np.array(self._preproc_data)
        # Start the background task with max_timeframe
        # The current timeframe will be used only for visualization
        self._bg_task.start(self.max_timeframe, gaze_data)
        self.on_update_ui()

    # Private - calculation callbacks

    def _on_bg_task_started(self):
        logger.debug("ScanPathController._on_bg_task_started")
        self._status_str = "Calculation started..."
        self._clear_data()
        self.on_update_ui()

    def _on_bg_task_updated(self, update_data):
        logger.debug("ScanPathController._on_bg_task_updated")
        item = ScanPathItem(update_data.frame_index, update_data.gaze_data)
        self._status_str = f"Calculation {int(self._bg_task.progress * 100)}%..."
        self._computed_storage.add(item)
        self.on_update_ui()

    def _on_bg_task_failed(self, error):
        logger.debug("ScanPathController._on_bg_task_failed")
        logger.error(f"Scan path calculation failed: {error}")
        self._status_str = "Calculation failed"
        self._clear_data()
        self.on_update_ui()

    def _on_bg_task_canceled(self):
        logger.debug("ScanPathController._on_bg_task_canceled")
        self._status_str = "Calculation canceled"
        self._clear_data()
        self.on_update_ui()

    def _on_bg_task_completed(self):
        logger.debug("ScanPathController._on_bg_task_completed")
        self._computed_storage.is_completed = True
        self._status_str = "Calculation completed"
        self.on_update_ui()
