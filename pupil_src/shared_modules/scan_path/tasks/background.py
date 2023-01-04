"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import namedtuple

import numpy as np
from background_helper import IPC_Logging_Task_Proxy
from observable import Observable
from scan_path.algorithm import ScanPathAlgorithm
from scan_path.utils import (
    SCAN_PATH_GAZE_DATUM_DTYPE,
    FakeGPool,
    generate_frames,
    scan_path_numpy_array_from,
    scan_path_zeros_numpy_array,
)

from .base import _BaseTask


class ScanPathBackgroundTask(Observable, _BaseTask):
    def __init__(self, g_pool):
        self.g_pool = g_pool
        self._bg_task = None
        self._progress = 0.0
        self._gaze_data = None

    # _BaseTask

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def is_active(self) -> bool:
        return self._bg_task is not None

    def start(self, timeframe, preprocessed_data):
        if self.is_active:
            return

        g_pool = FakeGPool(self.g_pool)

        self._gaze_data = scan_path_zeros_numpy_array()

        self._bg_task = IPC_Logging_Task_Proxy(
            "Scan path",
            generate_frames_with_corrected_gaze,
            args=(g_pool, timeframe, preprocessed_data),
        )

    def process(self):
        if self._bg_task:
            try:
                task_data = self._bg_task.fetch()
            except Exception as err:
                self._bg_task.cancel()
                self._bg_task = None
                self.on_failed(err)

            for progress, gaze_data in task_data:
                gaze_data = scan_path_numpy_array_from(gaze_data)
                self._gaze_data = np.append(self._gaze_data, gaze_data)
                self._progress = progress
                self.on_updated(gaze_data)

            if self._bg_task.completed:
                self._bg_task = None
                self._gaze_data = scan_path_numpy_array_from(self._gaze_data)
                self.on_completed(self._gaze_data)

    def cancel(self):
        if self._bg_task is not None:
            self._bg_task.cancel()
            self._bg_task = None
            self.on_canceled()
        self._progress = 0.0

    def cleanup(self):
        self.cancel()


def generate_frames_with_corrected_gaze(g_pool, timeframe, preprocessed_data):
    sp = ScanPathAlgorithm(timeframe)

    for progress, frame in generate_frames(g_pool):
        gaze_data = preprocessed_data[preprocessed_data.frame_index == frame.index]
        gaze_data = sp.update_from_frame(frame, gaze_data)
        yield progress, gaze_data
