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
from types import SimpleNamespace

from observable import Observable

from .base_task import _BaseTask
from .scan_path_utils import timestamp_ns, sec_to_ns, generate_frames_with_deserialized_gaze

class _BaseState:
    def __init__(self, g_pool):
        self.g_pool = g_pool


class UninitializedState(_BaseState):
    pass


class StartedState(_BaseState):
    pass


class ActiveState(_BaseState):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.generator = generate_frames_with_deserialized_gaze(g_pool)


class CompletedState(_BaseState):
    pass


class CanceledState(_BaseState):
    pass


PreprocessedGazeData = namedtuple("PreprocessedGazeData", ["frame", "gaze_data"])


class ScanPathPreprocessingTask(Observable, _BaseTask):

    def __init__(self, g_pool):
        self.g_pool = g_pool
        self._progress = 0.0
        self._state = UninitializedState(self.g_pool)

    # _BaseTask

    @property
    def is_active(self) -> bool:
        return isinstance(self._state, ActiveState)

    @property
    def progress(self) -> float:
        return self._progress

    def start(self):
        self._progress = 0.0
        self._state = StartedState(self.g_pool)
        self.on_started()

    def process(self, time_limit_sec: float = 0.01):
        time_limit_ns = sec_to_ns(time_limit_sec)

        if isinstance(self._state, (UninitializedState, CompletedState, CanceledState)):
            return

        if isinstance(self._state, StartedState):
            self._state = ActiveState(self.g_pool)

        assert isinstance(self._state, ActiveState)

        generator_is_done = True
        start_time_ns = timestamp_ns()

        for progress, current_frame, deserialized_gaze in self._state.generator:
            generator_is_done = False

            update_data = PreprocessedGazeData(current_frame, deserialized_gaze)

            self._progress = progress
            self.on_updated(update_data)

            time_diff_ns = timestamp_ns() - start_time_ns
            if time_diff_ns > time_limit_ns:
                break

        if generator_is_done:
            self._progress = 1.0
            self._state = CompletedState(self.g_pool)
            self.on_completed()

    def cancel(self):
        if isinstance(self._state, ActiveState):
            self._progress = 0.0
            self._state = CanceledState(self.g_pool)
            self.on_canceled()

    def cleanup(self):
        self.cancel()
