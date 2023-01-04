"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import copy
import functools

import cv2
import file_methods as fm
import methods as m
import numpy as np

from .utils import (
    np_denormalize,
    np_normalize,
    scan_path_numpy_array_from,
    scan_path_zeros_numpy_array,
)


class ScanPathAlgorithm:
    def __init__(self, timeframe: float):
        assert timeframe

        # user settings
        self.timeframe = timeframe

        # algorithm working data
        self.reset()

    def reset(self):
        self._prev_frame_index = -1
        self._prev_gray_image = None
        self._prev_gaze_datums = scan_path_zeros_numpy_array()

    def update_from_frame(self, frame, preprocessed_data):
        if frame.is_fake:
            self.reset()
            return scan_path_numpy_array_from(preprocessed_data)

        width, height = frame.width, frame.height
        return self.update_from_raw_data(
            frame_index=frame.index,
            preprocessed_data=preprocessed_data,
            image_size=(width, height),
            gray_image=frame.gray,
        )

    def update_from_raw_data(
        self, frame_index, preprocessed_data, image_size, gray_image
    ):
        if self._prev_frame_index + 1 != frame_index:
            self.reset()

        # lets update past gaze using optical flow: this is like sticking the gaze points onto the pixels of the img.
        if len(self._prev_gaze_datums) > 0:
            prev_gaze_points = np.zeros(
                (self._prev_gaze_datums.shape[0], 2), dtype=np.float32
            )
            prev_gaze_points[:, 0] = self._prev_gaze_datums["norm_x"]
            prev_gaze_points[:, 1] = self._prev_gaze_datums["norm_y"]
            prev_gaze_points = np_denormalize(prev_gaze_points, size=image_size)

            new_gaze_points, status, err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray_image,
                gray_image,
                prev_gaze_points,
                None,
                **self._lk_params,
            )

            new_gaze_points = np_normalize(new_gaze_points, size=image_size)

            new_gaze_data = scan_path_zeros_numpy_array(new_gaze_points.shape[0])
            new_gaze_data["timestamp"] = self._prev_gaze_datums["timestamp"]
            new_gaze_data["norm_x"] = new_gaze_points[:, 0]
            new_gaze_data["norm_y"] = new_gaze_points[:, 1]

            # Only keep gaze data where the status is 1
            status = np.array(status, dtype=bool)
            status.shape = -1  # flatten to keep dimensionality below
            new_gaze_data = new_gaze_data[status]
        else:
            new_gaze_data = scan_path_zeros_numpy_array()

        # trim gaze that is too old
        if len(preprocessed_data) > 0:
            now = preprocessed_data[0]["timestamp"]
            cutoff = now - self.timeframe
            new_gaze_data = new_gaze_data[new_gaze_data["timestamp"] > cutoff]

        # inject the scan path gaze points into recent_gaze_positions
        all_gaze_datums = np.concatenate([new_gaze_data, preprocessed_data])
        all_gaze_datums["frame_index"] = frame_index
        all_gaze_datums = np_sort_by_named_columns(all_gaze_datums, ["timestamp"])

        # update info for next frame.
        self._prev_gray_image = gray_image
        self._prev_frame_index = frame_index
        self._prev_gaze_datums = all_gaze_datums

        return all_gaze_datums

    # Private

    # vars for calcOpticalFlowPyrLK
    _lk_params = dict(
        winSize=(90, 90),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        minEigThreshold=0.005,
    )


def np_sort_by_named_columns(array, colums_by_priority):
    for col_name in reversed(colums_by_priority):
        array = array[array[col_name].argsort(kind="mergesort")]
    return array
