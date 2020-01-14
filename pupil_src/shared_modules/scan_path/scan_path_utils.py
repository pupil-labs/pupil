"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from types import SimpleNamespace

import numpy as np

from pupil_recording import PupilRecording
from video_capture.file_backend import File_Source, EndofVideoError
from gaze_producer.gaze_from_recording import GazeFromRecording
import methods as m
import player_methods as pm


_DESERIALIZED_GAZE_DTYPE = np.dtype([
    ('norm_x', np.float32),
    ('norm_y', np.float32),
    ('timestamp', np.float64),
])


class FakeGPool(SimpleNamespace):
    def __init__(self, g_pool):
        self.rec_dir = g_pool.rec_dir
        self.app = g_pool.app
        self.ipc_pub = g_pool.ipc_pub
        self.min_data_confidence = g_pool.min_data_confidence
        self.timestamps = g_pool.timestamps


def generate_frames_with_deserialized_gaze(g_pool):
    for progress, current_frame, gaze_datums in generate_frames_with_gaze(g_pool):
        deserialized_gaze = [(g["norm_pos"][0], g["norm_pos"][1], g["timestamp"]) for g in gaze_datums]
        deserialized_gaze = np.array(deserialized_gaze, dtype=_DESERIALIZED_GAZE_DTYPE)
        yield progress, current_frame, deserialized_gaze


def generate_frames_with_gaze(g_pool):
    for progress, current_frame in generate_frames(g_pool):
        frame_ts_window = pm.enclosing_window(g_pool.timestamps, current_frame.index)
        gaze_datums = g_pool.gaze_positions.by_ts_window(frame_ts_window)
        gaze_datums = [g for g in gaze_datums if g["confidence"] >= g_pool.min_data_confidence]

        yield progress, current_frame, gaze_datums


def generate_frames(g_pool):
    recording = PupilRecording(g_pool.rec_dir)
    video_path = recording.files().world()[0]

    fs = File_Source(g_pool, source_path=video_path)

    total_frame_count = fs.get_frame_count()

    while True:
        try:
            current_frame = fs.get_frame()
        except EndofVideoError:
            break

        progress = current_frame.index / total_frame_count

        yield progress, current_frame
