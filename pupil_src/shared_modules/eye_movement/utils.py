"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
import logging
import methods as mt
import file_methods as fm
import numpy as np


Gaze_Data = t.Iterable[fm.Serialized_Dict]

Gaze_Timestamp = float
Gaze_Time = t.Iterable[Gaze_Timestamp]

MsgPack_Serialized_Segment = t.Type[bytes]


logger = logging.getLogger(__name__)


EYE_MOVEMENT_TOPIC_PREFIX = "eye_movement."
EYE_MOVEMENT_EVENT_KEY = "eye_movement_segments"
EYE_MOVEMENT_GAZE_KEY = "eye_movement"


def can_use_3d_gaze_mapping(gaze_data) -> bool:
    return "gaze_normal_3d" in gaze_data[0] or "gaze_normals_3d" in gaze_data[0]


def clean_3d_data(gaze_points_3d: np.ndarray) -> np.ndarray:
    # Sometimes it is possible that the predicted gaze is behind the camera which is physically impossible.
    gaze_points_3d[gaze_points_3d[:, 2] < 0] *= -1.0
    return gaze_points_3d


def np_denormalize(points_2d, frame_size):
    width, height = frame_size
    points_2d[:, 0] *= width
    points_2d[:, 1] = (1.0 - points_2d[:, 1]) * height
    return points_2d


def gaze_data_to_nslr_data(capture, gaze_data, gaze_timestamps, use_pupil: bool):

    if use_pupil:
        assert can_use_3d_gaze_mapping(gaze_data)
        gaze_points_3d = [gp["gaze_point_3d"] for gp in gaze_data]
        gaze_points_3d = np.array(gaze_points_3d, dtype=np.float32)
        gaze_points_3d = clean_3d_data(gaze_points_3d)
    else:
        gaze_points_2d = np.array([gp["norm_pos"] for gp in gaze_data])
        gaze_points_2d = np_denormalize(gaze_points_2d, capture.frame_size)
        gaze_points_3d = capture.intrinsics.unprojectPoints(gaze_points_2d)

    x, y, z = gaze_points_3d.T
    r, theta, psi = mt.cart_to_spherical([x, y, z])

    angles = [theta, psi]
    angles = np.rad2deg(angles)

    nslr_data = np.column_stack(angles)

    validate_nslr_data(
        eye_positions=nslr_data,
        eye_timestamps=gaze_timestamps,
    )

    return nslr_data


def validate_nslr_data(eye_positions: np.ndarray, eye_timestamps: np.ndarray):
    def has_nan(arr: np.ndarray):
        return np.any(np.isnan(arr))

    def is_monotonic(arr: np.ndarray):
        return np.all(arr[:-1] <= arr[1:])

    def is_unique(arr: np.ndarray):
        return arr.shape == np.unique(arr, axis=0).shape

    if has_nan(eye_positions):
        raise ValueError("Gaze data contains NaN values")
    if not is_monotonic(eye_timestamps):
        raise ValueError("Gaze timestamps contain NaN values")
    if not is_monotonic(eye_timestamps):
        raise ValueError("Gaze timestamps are not monotonic")
    if not is_unique(eye_timestamps):
        raise ValueError("Gaze timestamps are not unique. Please recalculate gaze mapping with only 1 mapper enabled")
