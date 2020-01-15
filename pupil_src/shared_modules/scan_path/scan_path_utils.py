"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from time import monotonic
from types import SimpleNamespace

import numpy as np

from pupil_recording import PupilRecording
from video_capture.file_backend import File_Source, EndofVideoError
from gaze_producer.gaze_from_recording import GazeFromRecording
import methods as m
import player_methods as pm


SCAN_PATH_GAZE_DATUM_DTYPE = np.dtype([
    ("frame_index", np.int64),
    ("timestamp", np.float64),
    ("norm_x", np.float32),
    ("norm_y", np.float32),
])


def scan_path_zeros_numpy_array(n=0):
    new_array = np.zeros(n, dtype=SCAN_PATH_GAZE_DATUM_DTYPE)
    new_array = new_array.view(np.recarray)
    return new_array


def scan_path_numpy_array_from(it):
    if len(it) == 0:
        return scan_path_zeros_numpy_array()

    array = np.asarray(it)

    if array.dtype == SCAN_PATH_GAZE_DATUM_DTYPE:
        return array.view(np.recarray)

    assert len(array.shape) == 2
    assert array.shape[1] == len(SCAN_PATH_GAZE_DATUM_DTYPE)

    new_array = scan_path_zeros_numpy_array(array.shape[0])

    new_array["frame_index"] = array[:, 0]
    new_array["timestamp"] = array[:, 1]
    new_array["norm_x"] = array[:, 2]
    new_array["norm_y"] = array[:, 3]

    return new_array


class FakeGPool(SimpleNamespace):
    def __init__(self, g_pool):
        self.rec_dir = g_pool.rec_dir
        self.app = g_pool.app
        self.ipc_pub = g_pool.ipc_pub
        # self.ipc_pub = None
        self.min_data_confidence = g_pool.min_data_confidence
        self.timestamps = g_pool.timestamps


def timestamp_ns() -> int:
    """
    Returns a monotonic timestamp in nanoseconds.
    """
    return sec_to_ns(monotonic())


def sec_to_ns(sec: float) -> int:
    return int(sec * 10E9)


def ns_to_sec(ns: int) -> float:
    return float(ns) / 10E9


def generate_frame_indices_with_deserialized_gaze(g_pool):
    # TODO: Don't use generate_frames_with_gaze; Instead use VideoSet's lookup to get the number of frames/timestamps
    for progress, current_frame, gaze_datums in generate_frames_with_gaze(g_pool):
        deserialized_gaze = [(current_frame.index, g["timestamp"], g["norm_pos"][0], g["norm_pos"][1]) for g in gaze_datums]
        deserialized_gaze = scan_path_numpy_array_from(deserialized_gaze)
        yield progress, deserialized_gaze


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
