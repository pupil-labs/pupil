"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import functools
import logging
import re
import typing as T
from itertools import chain

import cv2
import file_methods as fm
import numpy as np

logger = logging.getLogger(__name__)


def enclosing_window(timestamps, idx):
    before = timestamps[idx - 1] if idx > 0 else -np.inf
    now = timestamps[idx]
    after = timestamps[idx + 1] if idx < len(timestamps) - 1 else np.inf
    return (now + before) / 2.0, (after + now) / 2.0


def exact_window(timestamps, index_range):
    end_index = min(index_range[1], len(timestamps) - 1)
    return (timestamps[index_range[0]], timestamps[end_index])


class Bisector:
    """Stores data with associated timestamps, both sorted by the timestamp."""

    def __init__(self, data=(), data_ts=()):
        if len(data) != len(data_ts):
            raise ValueError(
                "Each element in `data` requires a corresponding"
                " timestamp in `data_ts`"
            )
        elif not len(data):
            self.data = np.array([], dtype=object)
            self.data_ts = np.array([])
            self.sorted_idc = []
        else:
            self.data_ts = np.asarray(data_ts)
            self.data = np.asarray(data, dtype=object)

            # Find correct order once and reorder both lists in-place
            self.sorted_idc = np.argsort(self.data_ts)
            self.data_ts = self.data_ts[self.sorted_idc]
            self.data = self.data[self.sorted_idc]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} len={len(self)}>"

    def copy(self):
        copy = type(self)()
        copy.data = self.data.copy()
        copy.data_ts = self.data_ts.copy()
        copy.sorted_idc = self.sorted_idc.copy()
        return copy

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
        return bool(len(self.data))

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
        self.data = np.insert(self.data, insert_idx, datum)


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


class PupilTopic:
    WildcardKey = type(...)
    EyeIdFilterKey = T.Union[WildcardKey, int, str, T.Iterable[int], T.Iterable[str]]
    DetectorTagFilterKey = T.Union[WildcardKey, str, T.Iterable[str]]

    _FORMAT_STRING_V1 = "pupil.{eye_id}"
    _FORMAT_STRING_V2 = "pupil.{eye_id}.{detector_tag}"
    _FORMAT_STRING_LEGACY = "pupil_positions.{eye_id}"

    _MATCH_FORMAT_STRING_V1 = r"^pupil\.(?P<eye_id>{eye_id})$"
    _MATCH_FORMAT_STRING_V2 = (
        r"^pupil\.(?P<eye_id>{eye_id})\.(?P<detector_tag>{detector_tag})$"
    )
    _MATCH_FORMAT_STRING_LEGACY = r"^pupil_positions\.(?P<eye_id>{eye_id})$"

    _legacy_method_to_detector_tag = {
        "2d c++": "2d",
        "3d c++": "3d",
        "2d python": "2d",  # pre v0.7 format
    }

    @staticmethod
    def create(topic: str, pupil_datum: dict) -> str:
        regex_v1 = PupilTopic._match_regex_v1()
        match_v1 = re.match(regex_v1, topic)
        if match_v1:
            detector_tag = pupil_datum["method"]
            if detector_tag in PupilTopic._legacy_method_to_detector_tag:
                detector_tag = PupilTopic._legacy_method_to_detector_tag[detector_tag]
            return PupilTopic._FORMAT_STRING_V2.format(
                eye_id=match_v1.group("eye_id"),
                detector_tag=detector_tag,
            )
        regex_v2 = PupilTopic._match_regex_v2()
        match_v2 = re.match(regex_v2, topic)
        if match_v2:
            return PupilTopic._FORMAT_STRING_V2.format(
                eye_id=match_v2.group("eye_id"),
                detector_tag=match_v2.group("detector_tag"),
            )

        regex_legacy = PupilTopic._match_regex_legacy()
        match_legacy = re.match(regex_legacy, topic)
        if match_legacy:
            detector_tag = pupil_datum["method"]
            if detector_tag in PupilTopic._legacy_method_to_detector_tag:
                detector_tag = PupilTopic._legacy_method_to_detector_tag[detector_tag]
            return PupilTopic._FORMAT_STRING_V2.format(
                eye_id=match_legacy.group("eye_id"),
                detector_tag=detector_tag,
            )

        raise ValueError(f'Invalid pupil topic: "{topic}" datum={pupil_datum}')

    @staticmethod
    def match(topic: str, eye_id=None, detector_tag=None):
        eye_id = PupilTopic._canonical_subpattern(eye_id)
        detector_tag = PupilTopic._canonical_subpattern(detector_tag)
        regex_v2 = PupilTopic._match_regex_v2(eye_id=eye_id, detector_tag=detector_tag)
        return re.match(regex_v2, topic)

    @staticmethod
    def _canonical_subpattern(key) -> str:
        if isinstance(key, slice) and key != slice(None, None, None):
            raise ValueError("Only unconstrained slices (`:`, `::`) allowed")
        elif key is None or key is ... or isinstance(key, slice):
            return None
        elif isinstance(key, str) or isinstance(key, int):
            return str(key)
        else:
            return f'({"|".join(map(str, key))})'

    @staticmethod
    @functools.lru_cache(128)
    def _match_regex_v2(
        eye_id: T.Optional[str] = None, detector_tag: T.Optional[str] = None
    ):
        if eye_id is None:
            eye_id = "[01]"

        if detector_tag is None:
            detector_tag = r"[^\.]+"

        pattern = PupilTopic._MATCH_FORMAT_STRING_V2.format(
            eye_id=eye_id,
            detector_tag=detector_tag,
        )

        return re.compile(pattern)

    @staticmethod
    @functools.lru_cache(32)
    def _match_regex_v1(eye_id: T.Optional[str] = None):
        if eye_id is None:
            eye_id = "[01]"

        pattern = PupilTopic._MATCH_FORMAT_STRING_V1.format(
            eye_id=eye_id,
        )

        return re.compile(pattern)

    @staticmethod
    @functools.lru_cache(32)
    def _match_regex_legacy(eye_id: T.Optional[str] = None):
        if eye_id is None:
            eye_id = "[01]"

        pattern = PupilTopic._MATCH_FORMAT_STRING_LEGACY.format(
            eye_id=eye_id,
        )

        return re.compile(pattern)


class PupilDataBisector:
    def __init__(
        self,
        data: T.Optional[fm.PLData] = None,
        bisectors: T.Optional[T.Dict[str, Bisector]] = None,
    ):
        if bisectors is not None:
            self._bisectors = bisectors
        else:
            if data is None:
                data = fm.PLData([], [], [])
            self._bisectors = self._bisectors_from_data(data)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} topics={list(self._bisectors.keys())}>"

    def _bisectors_from_data(self, data: fm.PLData) -> T.Dict[str, Bisector]:
        _bisectors: T.Dict[str, Bisector] = {}
        for pupil_topic, data in self._group_data_by_pupil_topic(data).items():
            assert pupil_topic not in _bisectors
            assert len(data.topics) == len(data.data) == len(data.timestamps)
            bisector = Bisector(data.data, data.timestamps)
            _bisectors[pupil_topic] = bisector
        return _bisectors

    def init_dict_for_window(self, ts_window):
        init_dict = collections.defaultdict(list)
        for topic, bisector in self._bisectors.items():
            _init_dict = bisector.init_dict_for_window(ts_window)
            for key, values in _init_dict.items():
                init_dict[key].extend(values)
            topics = [topic] * len(_init_dict["data"])
            init_dict["topics"].extend(topics)

        return init_dict

    @staticmethod
    def from_init_dict(init_dict):
        data = fm.PLData(init_dict["data"], init_dict["data_ts"], init_dict["topics"])
        return PupilDataBisector(data)

    @functools.lru_cache(32)
    def __getitem__(
        self, key: T.Tuple[PupilTopic.EyeIdFilterKey, PupilTopic.DetectorTagFilterKey]
    ) -> Bisector:
        bisectors = [
            B for topic, B in self._bisectors.items() if PupilTopic.match(topic, *key)
        ]
        return self.combine_bisectors(bisectors)

    def by_ts_window(self, ts_window):
        bisectors = self._bisectors.values()
        init_dicts = [b.init_dict_for_window(ts_window) for b in bisectors]
        bisectors = [Bisector(**init_dict) for init_dict in init_dicts]
        combined = self.combine_bisectors(bisectors)
        return combined

    def by_ts(self, ts):
        # Returns datum for first bisector that contains it
        # TODO: Might require rework depending on where/how it is used
        for bisector in self._bisectors.values():
            try:
                return bisector.by_ts(ts)
            except ValueError:
                continue
        raise ValueError

    def __bool__(self):
        return any(self._bisectors.values())

    def __iter__(self):
        all_bisectors = self._bisectors.values()
        return iter(self.combine_bisectors(all_bisectors))

    @staticmethod
    def combine_bisectors(bisectors: T.Iterable[Bisector]) -> Bisector:
        data = list(chain.from_iterable(b.data for b in bisectors))
        data_ts = list(chain.from_iterable(b.data_ts for b in bisectors))
        return Bisector(data, data_ts)

    @classmethod
    def load_from_file(cls, dir_path, filename) -> "PupilDataBisector":
        data = fm.load_pldata_file(dir_path, filename)
        return cls(data=data)

    def save_to_file(self, dir_path, filename):
        with fm.PLData_Writer(dir_path, filename) as writer:
            for topic, bisector in self._bisectors.items():
                for timestamp, datum in zip(bisector.timestamps, bisector.data):
                    writer.append_serialized(timestamp, topic, datum.serialized)

    ### PRIVATE

    @staticmethod
    def _group_data_by_pupil_topic(data: fm.PLData) -> T.Dict[str, fm.PLData]:
        assert len(data.topics) == len(data.data) == len(data.timestamps)
        data_by_topic = collections.defaultdict(lambda: fm.PLData([], [], []))
        for raw_topic, datum, ts in zip(data.topics, data.data, data.timestamps):
            pupil_topic = PupilTopic.create(raw_topic, datum)
            data_by_topic[pupil_topic].data.append(datum)
            data_by_topic[pupil_topic].timestamps.append(ts)
            data_by_topic[pupil_topic].topics.append(raw_topic)
        return data_by_topic


class PupilDataCollector:
    def __init__(self):
        self._collection = collections.defaultdict(dict)

    def append(self, topic, datum, timestamp):
        pupil_topic = PupilTopic.create(topic, datum)
        self._collection[pupil_topic][timestamp] = datum

    def clear(self):
        self._collection.clear()

    def as_pupil_data_bisector(self) -> PupilDataBisector:
        bisectors = {}
        for topic, timestamps_data in self._collection.items():
            timestamps = list(timestamps_data.keys())
            data = list(timestamps_data.values())
            bisector = Bisector(data, timestamps)
            bisectors[topic] = bisector
        pupil_data_bisector = PupilDataBisector(bisectors=bisectors)
        return pupil_data_bisector

    def count_collected(self, eye_id=None, detector_tag=None):
        num_collected = 0
        for topic, values in self._collection.items():
            if PupilTopic.match(topic, eye_id, detector_tag):
                num_collected += len(values)
        return num_collected


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


def transparent_circle(img, center, radius, color, thickness):
    center = tuple(map(int, center))
    assert len(color) == 4 and all(type(c) == float for c in color)
    color = np.clip(color, 0.0, 1.0)  # sometimes the sliders returns values > 1.0
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
    except Exception:
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
