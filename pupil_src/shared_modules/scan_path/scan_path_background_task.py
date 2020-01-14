"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import namedtuple

from observable import Observable
from background_helper import IPC_Logging_Task_Proxy

from .base_task import _BaseTask
from .scan_path_utils import FakeGPool, generate_frames
from .scan_path_algorithm import ScanPathAlgorithm


CorrectedGazeData = namedtuple("CorrectedGazeData", ["frame_index", "gaze_data"])


class ScanPathBackgroundTask(Observable, _BaseTask):

    def __init__(self, g_pool):
        self.g_pool = g_pool
        self._bg_task = None
        self._progress = 0.0

    # _BaseTask

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def is_active(self) -> bool:
        return self._bg_task is not None

    def start(self, timeframe, deserialized_gaze):
        if self.is_active:
            return

        g_pool = FakeGPool(self.g_pool)

        self._bg_task = IPC_Logging_Task_Proxy(
            "Scan path",
            generate_frames_with_corrected_gaze,
            args=(g_pool, timeframe, deserialized_gaze),
        )

    def process(self):
        if self._bg_task:
            try:
                task_data = self._bg_task.fetch()
            except Exception as err:
                self._bg_task.cancel()
                self._bg_task = None
                self.on_failed(err)

            for progress, frame_index, gaze_data in task_data:
                update_data = CorrectedGazeData(frame_index, gaze_data)
                self._progress = progress
                self.on_updated(update_data)

            if self._bg_task.completed:
                self._bg_task = None
                self.on_completed()

    def cancel(self):
        if self._bg_task is not None:
            self._bg_task.cancel()
            self._bg_task = None
            self._progress = 0.0
            self.on_canceled()

    def cleanup(self):
        self.cancel()


def generate_frames_with_corrected_gaze(g_pool, timeframe, deserialized_gaze):
    sp = ScanPathAlgorithm(timeframe)

    for progress, frame in generate_frames(g_pool):
        gaze_datums = deserialized_gaze[frame.index]
        corrected_gaze_datums = sp.update_from_frame(frame, gaze_datums)
        yield progress, frame.index, corrected_gaze_datums
