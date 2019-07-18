"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import os

import cv2
import numpy as np

import csv_utils

logger = logging.getLogger(__name__)


def enclosing_window(timestamps, idx):
    before = timestamps[idx - 1] if idx > 0 else -np.inf
    now = timestamps[idx]
    after = timestamps[idx + 1] if idx < len(timestamps) - 1 else np.inf
    return (now + before) / 2.0, (after + now) / 2.0


def exact_window(timestamps, index_range):
    end_index = min(index_range[1], len(timestamps) - 1)
    return (timestamps[index_range[0]], timestamps[end_index])


class Bisector(object):
    """Stores data with associated timestamps, both sorted by the timestamp."""

    def __init__(self, data=(), data_ts=()):
        if len(data) != len(data_ts):
            raise ValueError(
                (
                    "Each element in `data` requires a corresponding"
                    " timestamp in `data_ts`"
                )
            )
        elif not data:
            self.data = []
            self.data_ts = np.asarray([])
            self.sorted_idc = []
        else:
            self.data_ts = np.asarray(data_ts)
            self.data = np.asarray(data, dtype=object)

            # Find correct order once and reorder both lists in-place
            self.sorted_idc = np.argsort(self.data_ts)
            self.data_ts = self.data_ts[self.sorted_idc]
            self.data = self.data[self.sorted_idc].tolist()

    def by_ts(self, ts):
        """
        :param ts: timestamp to extract.
        :return: datum that is matching
        :raises: ValueError if no matching datum is found
        """
        found_index = np.searchsorted(self.data_ts, ts)
        try:
            found_data = self.data[found_index]
            found_ts = self.data_ts[found_index]
        except IndexError:
            raise ValueError
        found = found_ts == ts
        if not found:
            raise ValueError
        else:
            return found_data

    def by_ts_window(self, ts_window):
        start_idx, stop_idx = self._start_stop_idc_for_window(ts_window)
        return self.data[start_idx:stop_idx]

    def _start_stop_idc_for_window(self, ts_window):
        return np.searchsorted(self.data_ts, ts_window)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __bool__(self):
        return bool(self.data)

    @property
    def timestamps(self):
        return self.data_ts

    def init_dict_for_window(self, ts_window):
        start_idx, stop_idx = self._start_stop_idc_for_window(ts_window)
        return {
            "data": self.data[start_idx:stop_idx],
            "data_ts": self.data_ts[start_idx:stop_idx],
        }


class Mutable_Bisector(Bisector):
    def insert(self, timestamp, datum):
        insert_idx = np.searchsorted(self.data_ts, timestamp)
        self.data_ts = np.insert(self.data_ts, insert_idx, timestamp)
        self.data.insert(insert_idx, datum)


class Affiliator(Bisector):
    """docstring for ClassName"""

    def __init__(self, data=(), start_ts=(), stop_ts=()):
        super().__init__(data, start_ts)
        self.stop_ts = np.asarray(stop_ts)
        self.stop_ts = self.stop_ts[self.sorted_idc]

    def _start_stop_idc_for_window(self, ts_window):
        start_idx = np.searchsorted(self.stop_ts, ts_window[0])
        stop_idx = np.searchsorted(self.data_ts, ts_window[1])
        return start_idx, stop_idx

    def init_dict_for_window(self, ts_window):
        start_idx, stop_idx = self._start_stop_idc_for_window(ts_window)
        return {
            "data": self.data[start_idx:stop_idx],
            "start_ts": self.data_ts[start_idx:stop_idx],
            "stop_ts": self.stop_ts[start_idx:stop_idx],
        }


def find_closest(target, source):
    """Find indeces of closest `target` elements for elements in `source`.
    -
    `source` is assumed to be sorted. Result has same shape as `source`.
    Implementation taken from:
    -
    https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python/8929827#8929827
    """
    target = np.asarray(target)  # fixes https://github.com/pupil-labs/pupil/issues/1439
    idx = np.searchsorted(target, source)
    idx = np.clip(idx, 1, len(target) - 1)
    left = target[idx - 1]
    right = target[idx]
    idx -= source - left < right - source
    return idx


def correlate_data(data, timestamps):
    """
    data:  list of data :
        each datum is a dict with at least:
            timestamp: float

    timestamps: timestamps list to correlate  data to

    this takes a data list and a timestamps list and makes a new list
    with the length of the number of timestamps.
    Each slot contains a list that will have 0, 1 or more assosiated data points.

    Finally we add an index field to the datum with the associated index
    """
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    data.sort(key=lambda d: d["timestamp"])

    while True:
        try:
            datum = data[data_index]
            # we can take the midpoint between two frames in time: More appropriate for SW timestamps
            ts = (timestamps[frame_idx] + timestamps[frame_idx + 1]) / 2.0
            # or the time of the next frame: More appropriate for Sart Of Exposure Timestamps (HW timestamps).
            # ts = timestamps[frame_idx+1]
        except IndexError:
            # we might loose a data point at the end but we dont care
            break

        if datum["timestamp"] <= ts:
            # datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index += 1
        else:
            frame_idx += 1

    return data_by_frame


def load_meta_info(rec_dir):
    meta_info_path = os.path.join(rec_dir, "info.csv")
    with open(meta_info_path, "r", encoding="utf-8") as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
    return meta_info


def is_pupil_rec_dir(rec_dir):
    if not os.path.isdir(rec_dir):
        logger.error("No valid dir supplied ({})".format(rec_dir))
        return False
    try:
        meta_info = load_meta_info(rec_dir)
        meta_info["Recording Name"]  # Test key existence
    except:
        logger.error("Could not read info.csv file: Not a valid Pupil recording.")
        return False
    return True


def transparent_circle(img, center, radius, color, thickness):
    center = tuple(map(int, center))
    assert len(color) == 4 and all(type(c) == float and 0.0 <= c <= 1.0 for c in color)
    bgr = [255 * c for c in color[:3]]  # convert to 0-255 scale for OpenCV
    alpha = color[-1]
    radius = int(radius)
    if thickness > 0:
        pad = radius + 2 + thickness
    else:
        pad = radius + 3
    roi = (
        slice(center[1] - pad, center[1] + pad),
        slice(center[0] - pad, center[0] + pad),
    )

    try:
        overlay = img[roi].copy()
        cv2.circle(img, center, radius, bgr, thickness=thickness, lineType=cv2.LINE_AA)
        opacity = alpha
        cv2.addWeighted(
            src1=img[roi],
            alpha=opacity,
            src2=overlay,
            beta=1.0 - opacity,
            gamma=0,
            dst=img[roi],
        )
    except:
        logger.debug(
            "transparent_circle would have been partially outside of img. Did not draw it."
        )


def transparent_image_overlay(pos, overlay_img, img, alpha):
    """
    Overlay one image with another with alpha blending
    In player this will be used to overlay the eye (as overlay_img) over the world image (img)
    Arguments:
        pos: (x,y) position of the top left corner in numpy row,column format from top left corner (numpy coord system)
        overlay_img: image to overlay
        img: destination image
        alpha: 0.0-1.0
    """
    roi = (
        slice(pos[1], pos[1] + overlay_img.shape[0]),
        slice(pos[0], pos[0] + overlay_img.shape[1]),
    )
    try:
        cv2.addWeighted(overlay_img, alpha, img[roi], 1.0 - alpha, 0, img[roi])
    except (TypeError, cv2.error):
        logger.debug(
            "transparent_image_overlay was outside of the world image and was not drawn"
        )
