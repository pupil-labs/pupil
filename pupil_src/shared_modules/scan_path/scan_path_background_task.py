"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
from types import SimpleNamespace

from observable import Observable
from background_helper import IPC_Logging_Task_Proxy
from video_capture.file_backend import File_Source, EndofVideoError
from gaze_producer.gaze_from_recording import GazeFromRecording
import methods as m
import player_methods as pm

from .scan_path_algorithm import ScanPathAlgorithm


class ScanPathBackgroundTask(Observable):

    def __init__(self, g_pool):
        self.g_pool = g_pool
        self._bg_task = None

    def start(self, timeframe):
        self.cleanup()

        g_pool = SimpleNamespace()
        g_pool.rec_dir = self.g_pool.rec_dir
        g_pool.app = self.g_pool.app
        g_pool.ipc_pub = self.g_pool.ipc_pub
        g_pool.min_data_confidence = self.g_pool.min_data_confidence

        self._bg_task = IPC_Logging_Task_Proxy(
            "Scan path",
            generate_frames_with_corrected_gaze,
            args=(g_pool, timeframe),
        )

    @property
    def is_running(self) -> bool:
        return self._bg_task is not None

    def process(self):
        if self._bg_task:
            try:
                task_data = self._bg_task.fetch()
            except Exception as err:
                self._bg_task.cancel()
                self._bg_task = None
                self.on_task_failed(err)

            for data in task_data:
                self.on_task_updated(*data)

            if self._bg_task.completed:
                self._bg_task = None
                self.on_task_completed()

    def cleanup(self):
        if self._bg_task is not None:
            self._bg_task.cancel()
            self._bg_task = None

    def on_task_started(self):
        pass

    def on_task_updated(self, progress, frame_index, gaze_datums, corrected_gaze_datums):
        pass

    def on_task_failed(self, error):
        pass

    def on_task_completed(self):
        pass


def generate_frames_with_corrected_gaze(g_pool, timeframe):
    sp = ScanPathAlgorithm(timeframe)

    for progress, frame, gaze_datums in generate_frames_with_gaze(g_pool):
        corrected_gaze_datums = sp.update_from_frame(frame, gaze_datums)
        yield progress, frame.index, gaze_datums, corrected_gaze_datums


def generate_frames_with_gaze(g_pool):

    video_path = os.path.join(g_pool.rec_dir, "world.mp4") #TODO: Use PupilRecording

    fs = File_Source(g_pool, source_path=video_path)
    
    gp = GazeFromRecording(g_pool)

    total_frame_count = fs.get_frame_count()

    while True:
        try:
            current_frame = fs.get_frame()
        except EndofVideoError:
            break

        progress = current_frame.index / total_frame_count

        frame_ts_window = pm.enclosing_window(g_pool.gaze_positions.timestamps, current_frame.index)
        gaze_datums = g_pool.gaze_positions.by_ts_window(frame_ts_window)
        gaze_datums = [g for g in gaze_datums if g["confidence"] >= g_pool.min_data_confidence]

        yield progress, current_frame, gaze_datums
