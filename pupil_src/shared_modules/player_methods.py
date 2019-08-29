"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import glob
import logging
import os
from pathlib import Path
import re
import typing as T

import cv2
import numpy as np

import csv_utils
from video_capture.utils import VIDEO_EXTS as VALID_VIDEO_EXTENSIONS


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
    return Pupil_Recording(rec_dir).meta_info


def is_pupil_rec_dir(rec_dir):
    try:
        Pupil_Recording(rec_dir)
        return True
    except InvalidRecordingException as e:
        logger.error(str(e))
        return False


class InvalidRecordingException(Exception):
    def __init__(self, reason: str, recovery: str = ""):
        message = (reason + "\n" + recovery) if recovery else reason
        super().__init__(message)
        self.reason = reason
        self.recovery = recovery

    def __str__(self):
        return f"{type(self).__name__}: {super().__str__()}"


class Pupil_Recording:
    def __init__(self, rec_dir):
        self._rec_dir = rec_dir
        self._meta_info = None
        self.load(rec_dir=rec_dir)

    @property
    def rec_dir(self):
        return self._rec_dir

    @property
    def meta_info(self):
        return self._meta_info

    @property
    def capture_software(self) -> str:
        return self.meta_info.get("Capture Software", "Pupil Capture")

    @property
    def data_format_version(self) -> T.Optional[str]:
        return self.meta_info.get("Data Format Version", None)

    @property
    def is_pupil_mobile(self) -> bool:
        return self.capture_software == "Pupil Mobile"

    @property
    def is_pupil_invisible(self) -> bool:
        return self.capture_software == "Pupil Invisible"

    def reload(self):
        self.load(rec_dir=self.rec_dir)

    def load(self, rec_dir):
        def normalize_extension(ext: str) -> str:
            if ext.startswith("."):
                ext = ext[1:]
            return ext

        def is_video_file(file_path):
            if not os.path.isfile(file_path):
                return False
            _, ext = os.path.splitext(file_path)
            ext = normalize_extension(ext)
            valid_video_extensions = map(normalize_extension, VALID_VIDEO_EXTENSIONS)
            if ext not in valid_video_extensions:
                return False
            return True

        if not os.path.exists(rec_dir):
            raise InvalidRecordingException(
                reason=f"Target at path does not exist: {rec_dir}", recovery=""
            )

        if not os.path.isdir(rec_dir):
            if is_video_file(rec_dir):
                raise InvalidRecordingException(
                    reason=f"The provided path is a video, not a recording directory",
                    recovery="Please provide a recording directory",
                )
            else:
                raise InvalidRecordingException(
                    reason=f"Target at path is not a directory: {rec_dir}", recovery=""
                )

        info_path = os.path.join(rec_dir, "info.csv")

        if not os.path.exists(info_path):
            raise InvalidRecordingException(
                reason=f"There is no info.csv in the target directory", recovery=""
            )

        if not os.path.isfile(info_path):
            raise InvalidRecordingException(
                reason=f"Target info.csv is not a file: {info_path}", recovery=""
            )

        with open(info_path, "r", encoding="utf-8") as csvfile:
            try:
                meta_info = csv_utils.read_key_value_file(csvfile)
            except Exception as e:
                raise InvalidRecordingException(
                    reason=f"Failed reading info.csv: {e}", recovery=""
                )

        info_mandatory_keys = ["Recording Name"]

        for key in info_mandatory_keys:
            try:
                meta_info[key]
            except KeyError:
                raise InvalidRecordingException(
                    reason=f'Target info.csv does not have "{key}"', recovery=""
                )

        all_file_paths = glob.iglob(os.path.join(rec_dir, "*"))

        # TODO: Should this validation be "are there any video files" or are there specific video files?
        if not any(is_video_file(path) for path in all_file_paths):
            raise InvalidRecordingException(
                reason=f"Target directory does not contain any video files", recovery=""
            )

        # TODO: Are there any other validations missing?
        # All validations passed

        self._rec_dir = rec_dir
        self._meta_info = meta_info

    class FileFilter(collections.Sequence):
        """Utility class for conveniently filtering files of the recording.

        Filters can be applied sequentially, since they return a filter again.
        Overloading __getitem__ and __len__ allows for full sequence functionality.
        Example usage:

            # prints all world videos in my_rec_dir
            for path in FileFilter("my_rec_dir").videos().world():
                print(f"World video file: {path}")
        """

        def world(self) -> "Pupil_Recording.FileFilter":
            patterns = [
                "world",  # pupil core
                "Pupil Cam([0-2]) ID2",  # pupil mobile
                "PI world v1 ps",  # PI
            ]
            return self.filter(*patterns)

        def videos(self) -> "Pupil_Recording.FileFilter":
            return self.filter(*VALID_VIDEO_EXTENSIONS)

        def raw_time(self) -> "Pupil_Recording.FileFilter":
            return self.filter(r"\.time")

        def mobile(self) -> "Pupil_Recording.FileFilter":
            return self.filter(r"Pupil Cam[0-3] ID[0-2]")

        # TODO: add more filters for other types of files

        def __init__(self, rec_dir: str):
            """Init filter to all files in the directory."""
            self.__files = [path for path in Path(rec_dir).iterdir() if path.is_file()]

        def __getitem__(self, key):
            # Used for implementing collections.Sequence
            return self.__files[key]

        def __len__(self):
            # Used for implementing collections.Sequence
            return len(self.__files)

        def filter(self, *patterns: str) -> "Pupil_Recording.FileFilter":
            """Filters current files, keeping anything matching any of the patterns."""
            self.__files = [
                item
                for item in self.__files
                if any(re.search(pattern, str(item)) for pattern in patterns)
            ]
            return self

    def files(self) -> "Pupil_Recording.FileFilter":
        return Pupil_Recording.FileFilter(self.rec_dir)


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
