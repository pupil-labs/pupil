"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import glob
import logging
import os
import pathlib as pl
from pathlib import Path
from typing import Iterator, Sequence

import av
import cv2
import numpy as np
from timestamp import legacy_timestamps_file_path_like


logger = logging.getLogger(__name__)

VIDEO_EXTS = ("mp4", "mjpeg", "h264", "mkv", "avi", "fake")
VIDEO_TIME_EXTS = VIDEO_EXTS + ("time",)


class Exposure_Time(object):
    def __init__(self, max_ET, frame_rate, mode="manual"):
        self.mode = mode
        self.ET_thres = 1, min(10000 / frame_rate, max_ET)
        self.last_ET = self.ET_thres[1]

        self.targetY_thres = 90, 150

        self.AE_Win = np.array(
            [
                [3, 1, 1, 1, 1, 1, 1, 3],
                [3, 1, 1, 1, 1, 1, 1, 3],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [3, 1, 1, 1, 1, 1, 1, 3],
                [3, 1, 1, 1, 1, 1, 1, 3],
            ]
        )
        self.smooth = 1 / 3
        self.check_freq = 0.1 / 3
        self.last_check_timestamp = None

    def calculate_based_on_frame(self, frame):
        if self.last_check_timestamp is None:
            self.last_check_timestamp = frame.timestamp

        if frame.timestamp - self.last_check_timestamp > self.check_freq:
            if self.mode == "manual":
                self.last_ET = self.ET_thres[1]
                return self.ET_thres[1]
            elif self.mode == "auto":
                image_block = cv2.resize(frame.gray, dsize=self.AE_Win.shape)
                YTotal = max(
                    np.multiply(self.AE_Win, image_block).sum() / self.AE_Win.sum(), 1
                )

                if YTotal < self.targetY_thres[0]:
                    targetET = self.last_ET * self.targetY_thres[0] / YTotal
                elif YTotal > self.targetY_thres[1]:
                    targetET = self.last_ET * self.targetY_thres[1] / YTotal
                else:
                    targetET = self.last_ET

                next_ET = np.clip(
                    self.last_ET + (targetET - self.last_ET) * self.smooth,
                    self.ET_thres[0],
                    self.ET_thres[1],
                )
                self.last_ET = next_ET
                return next_ET


class Check_Frame_Stripes(object):
    def __init__(
        self,
        check_freq_init=0.1,
        check_freq_upperbound=5,
        check_freq_lowerbound=0.00001,
        factor=0.8,
    ):
        self.check_freq_init = check_freq_init
        self.check_freq = self.check_freq_init
        self.check_freq_upperbound = check_freq_upperbound
        self.check_freq_lowerbound = check_freq_lowerbound
        self.factor = factor
        self.factor_mul = 1.1

        self.last_check_timestamp = None
        self.len_row_index = 8
        self.len_col_index = 8
        self.row_index = None
        self.col_index = None

    def require_restart(self, frame):
        if self.last_check_timestamp is None:
            self.last_check_timestamp = frame.timestamp

        if frame.timestamp - self.last_check_timestamp > self.check_freq:
            self.last_check_timestamp = frame.timestamp
            res = self.check_slice(frame.gray)

            if res is True:
                self.check_freq = (
                    min(self.check_freq_init, self.check_freq) * self.factor
                )
                if self.check_freq < self.check_freq_lowerbound:
                    return True
            elif res is False:
                if self.check_freq < self.check_freq_upperbound:
                    self.check_freq = min(
                        self.check_freq * self.factor_mul, self.check_freq_upperbound
                    )

        return False

    def check_slice(self, frame_gray):
        num_local_optimum = [0, 0]
        if self.row_index is None:
            self.row_index = np.linspace(
                8, frame_gray.shape[0] - 8, num=self.len_row_index, dtype=np.int
            )
            self.col_index = np.linspace(
                8, frame_gray.shape[1] - 8, num=self.len_col_index, dtype=np.int
            )
        for n in [0, 1]:
            if n == 0:
                arrs = np.array(frame_gray[self.row_index, :], dtype=np.int)
            else:
                arrs = np.array(frame_gray[:, self.col_index], dtype=np.int)
                arrs = np.transpose(arrs)

            local_max_union = set()
            local_min_union = set()
            for arr in arrs:
                local_max = set(
                    np.where(
                        np.r_[False, True, arr[2:] > arr[:-2] + 30]
                        & np.r_[arr[:-2] > arr[2:] + 30, True, False]
                        is True
                    )[0]
                )
                local_min = set(
                    np.where(
                        np.r_[False, True, arr[2:] + 30 < arr[:-2]]
                        & np.r_[arr[:-2] + 30 < arr[2:], True, False]
                        is True
                    )[0]
                )
                num_local_optimum[n] += len(
                    local_max_union.intersection(local_max)
                ) + len(local_min_union.intersection(local_min))
                if sum(num_local_optimum) >= 3:
                    return True
                local_max_union = local_max_union.union(local_max)
                local_min_union = local_min_union.union(local_min)

        if sum(num_local_optimum) == 0:
            return False
        else:
            return None


class Video:
    def __init__(self, path: str) -> None:
        self.path = path
        self.ts = None
        self._pts = None
        self._is_valid = self.check_valid()

    @property
    def is_valid(self):
        return self._is_valid

    def check_valid(self):
        try:
            cont = av.open(self.path)
            # Three failure scenarios:
            # 1. Broken video -> AVError
            # 2. decode() does not yield anything
            # 3. decode() yields None
            first_frame = next(cont.decode(video=0), None)
            if first_frame is None:
                raise av.AVError("Video does not contain any frames")
        except av.AVError:
            return False
        else:
            cont.seek(0)
            self.cont = cont
            return True

    def load_valid_container(self):
        if self.is_valid:
            return self.cont
        else:
            return None

    def load_ts(self):
        try:
            self.ts = np.load(self.ts_loc)
        except FileNotFoundError:
            self.ts = np.array([])
            return
        self.ts = self._fix_negative_time_jumps(self.ts)

    def load_pts(self):
        packets = self.cont.demux(video=0)
        # last pts is invalid
        self._pts = np.array([packet.pts for packet in packets][:-1])

    @property
    def name(self) -> str:
        file_ = os.path.split(self.path)[1]
        return os.path.splitext(file_)[0]

    @property
    def ts_loc(self) -> str:
        return legacy_timestamps_file_path_like(os.path.join(self.base, self.name))

    @property
    def base(self) -> str:
        return os.path.split(self.path)[0]

    @property
    def timestamps(self) -> np.ndarray:
        if self.ts is None:
            self.load_ts()
        return self.ts

    @property
    def pts(self) -> np.ndarray:
        if self._pts is None:
            self.load_pts()
        return self._pts

    @staticmethod
    def _fix_negative_time_jumps(timestamps: np.ndarray) -> np.ndarray:
        """Fix cases when large negative time jumps cause huge gaps due to sorting

        Replaces timestamps causing negative jumps with mean value of its adjacent
        timestamps. This work-around is based on the assumption that the negative time
        jump is caused by a single invalid timestamp.

        Work around for https://github.com/pupil-labs/pupil/issues/1550
        """
        # TODO: what if adjacent timestamps are negative/zero as well?
        time_diff = np.diff(timestamps)
        invalid_idc = np.flatnonzero(time_diff < 0)
        timestamps[invalid_idc + 1] = np.mean(
            (timestamps[invalid_idc + 2], timestamps[invalid_idc]), axis=0
        )
        return timestamps


class LookupTableNotInitializedError(AttributeError):
    pass


class VideoSet:
    def __init__(self, rec: str, name: str, fill_gaps: bool):
        self.rec = rec
        self.name = name
        self.fill_gaps = fill_gaps
        self.video_exts = set(VIDEO_EXTS) - {"fake"}
        self._videos = sorted(self.fetch_videos(), key=lambda v: v.path)
        self._containers = [vid.load_valid_container() for vid in self.videos]

    def is_empty(self) -> bool:
        try:
            # bool(self.lookup.timestamp) raises ValueError for numpy arrays: The truth
            # value of an array with more than one element is ambiguous.
            return len(self.lookup.timestamp) == 0
        except AttributeError:
            raise LookupTableNotInitializedError(
                "Lookup table was not initialized correctly!"
            )

    @property
    def videos(self) -> Sequence[Video]:
        return self._videos

    @property
    def containers(self) -> Sequence[Video]:
        return self._containers

    @property
    def lookup_loc(self) -> str:
        return os.path.join(self.rec, f"{self.name}_lookup.npy")

    def fetch_videos(self) -> Iterator[Video]:
        # If self.fill_gaps, we return all videos
        # else we skip the broken videos
        for ext in self.video_exts:
            for loc in Path(self.rec).glob(f"{self.name}*.{ext}"):
                vid = Video(str(loc))
                if self.fill_gaps or vid.is_valid:
                    yield vid

    def build_lookup(self, fallback_timestamps=None):
        """
        The lookup table is a np.recarray containing entries
        for each (virtual and real) frame.

        Each entry consists of 3 values:
            - container_idx: Corresponding self.videos index
            - container_frame_idx: Frame index within the container
            - timestamp: Recorded or virtual Pupil timestamp

        container_idx entries of value -1 indicate a virtual frame.

        The lookup table can be easiliy filtered for real frames:
            lookup = lookup[lookup.container_idx > -1]

        Use case:
        Given a Pupil timestamp, one can use bisect to find the corresponding
        lookup entry index. From there, one can lookup the corresponding
        container, load it if necessary, and calculate the target PTS

        Case 1: all videos are valid and self._fill_gaps is True
            base case
        Case 2: all videos are valid and self._fill_gaps is False
            skip to the next video, use for detection
        Case 3: some videos are broken and self._fill_gaps is True
            return gray frame for the broken video
        Case 4: some videos are broken and self._fill_gaps is False
            skip to the next valid video, use for detection
        Case 5: all videos are broken and self._fill_gaps is True
            return gray frame for the broken video
        Case 6: all videos are broken and self._fill_gaps is False
            return
        """

        loaded_ts = self._loaded_ts_sorted()
        loaded_ts = self._fill_gaps(loaded_ts)

        if len(loaded_ts) == 0 and fallback_timestamps is not None:
            fallback_timestamps = np.asanyarray(fallback_timestamps)
            self.lookup = self._setup_lookup(fallback_timestamps)
            return

        lookup = self._setup_lookup(loaded_ts)
        for container_idx, vid in enumerate(self.videos):
            if not vid.is_valid:
                # For invalid videos, we still try to load the timestamps (might be empty)
                lookup_mask = np.isin(lookup.timestamp, vid.timestamps)
                lookup.container_frame_idx[lookup_mask] = np.arange(vid.timestamps.size)
            else:
                # NOTE: For unknown reasons we sometimes have more timestamps than
                # frames. We don't know how to match non-matching timestamps and
                # pts, so we might introduce a systematic bias when fixing this! The
                # idea is to keep only data for timestamps that were recorded, but
                # leave frames blank if we don't have frame information.
                npts = vid.pts.size
                ntime = vid.timestamps.size
                if npts < ntime:
                    logger.warning(
                        f"Found {ntime} timestamps vs {npts} frames!"
                        f" Last {abs(npts - ntime)} frames are empty!"
                    )
                elif ntime < npts:
                    logger.warning(
                        f"Found {ntime} timestamps vs {npts} frames!"
                        f" Discarding last {abs(npts - ntime)} frames!"
                    )
                data_size = min(npts, ntime)
                vid_timestamps = vid.timestamps[:data_size]
                vid_pts = vid.pts[:data_size]

                lookup_mask = np.isin(lookup.timestamp, vid_timestamps)
                lookup.container_frame_idx[lookup_mask] = np.arange(vid_timestamps.size)
                lookup.container_idx[lookup_mask] = container_idx
                lookup.pts[lookup_mask] = vid_pts
        self.lookup = lookup
        np.save(self.lookup_loc, self.lookup)
        # filter gaps (after saving!)
        if not self.fill_gaps:
            self._remove_filled_gaps()

    def load_lookup(self):
        self.lookup = np.load(self.lookup_loc).view(np.recarray)
        if not self.fill_gaps:
            self._remove_filled_gaps()

    def load_or_build_lookup(self):
        try:
            self.load_lookup()
        except FileNotFoundError:
            self.build_lookup()

    def _loaded_ts_sorted(self) -> np.ndarray:
        if not self.videos:
            return np.array([])
        loaded_ts = [vid.timestamps for vid in self.videos]
        all_ts = np.concatenate(loaded_ts)
        return all_ts

    def _remove_filled_gaps(self):
        cont_idc = self.lookup.container_idx
        self.lookup = self.lookup[cont_idc > -1]

    def _fill_gaps(self, timestamps: np.ndarray) -> np.ndarray:
        time_diff = np.diff(timestamps)
        if time_diff.size > 0:
            median_time_diff = np.median(time_diff)
        else:
            # TODO: Not sure if this is an acceptable value, but this is what np.median returns for an empty input
            median_time_diff = np.nan
        gap_start_idc = np.flatnonzero(
            time_diff > self._gap_fill_threshold(median_time_diff)
        )
        gap_stop_idc = gap_start_idc + 1
        gap_fill_starts = timestamps[gap_start_idc] + median_time_diff
        gap_fill_stops = timestamps[gap_stop_idc] - median_time_diff

        all_ts = [timestamps]
        for start, stop in zip(gap_fill_starts, gap_fill_stops):
            all_ts.append(np.arange(start, stop, median_time_diff))

        all_ts = np.concatenate(all_ts)
        all_ts.sort()
        return all_ts

    def _gap_fill_threshold(self, median=0.03):
        """
        Frame timestamp difference [seconds] that needs to be exceeded
        in order to start filling frames.

        median: Median frame timestamp difference in seconds

        return: float [seconds], should be >= median
        """
        return max(1.0, median)  # return e.g. 4 * median for dynamic gap filling

    def _setup_lookup(self, timestamps: np.ndarray) -> np.recarray:
        lookup_entry = np.dtype(
            [
                ("container_idx", "<i8"),
                ("container_frame_idx", "<i8"),
                ("timestamp", "<f8"),
                ("pts", "<i8"),
            ]
        )
        lookup = np.empty(timestamps.size, dtype=lookup_entry).view(np.recarray)
        lookup.timestamp = timestamps
        lookup.container_idx = -1  # virtual container by default
        return lookup


def pi_gaze_items(root_dir):
    def find_raw_path(timestamps_path):
        raw_name = timestamps_path.name.replace("_timestamps", "")
        raw_path = timestamps_path.with_name(raw_name).with_suffix(".raw") #TODO: legacy_timestamps_file_path_without_suffix
        assert raw_path.exists(), f"The file does not exist at path: {raw_path}"
        return raw_path

    def load_timestamps_data(path):
        timestamps = np.load(str(path))
        return timestamps

    def load_raw_data(path):
        raw_data = np.fromfile(str(path), "<f4")
        raw_data_dtype = raw_data.dtype
        raw_data.shape = (-1, 2)
        return np.asarray(raw_data, dtype=raw_data_dtype)

    # This pattern will match any filename that:
    # - starts with "gaze ps"
    # - is followed by one or more digits
    # - is followed by "_timestamps.npy"
    gaze_timestamp_pattern = "gaze ps[0-9]*_timestamps.npy" # TODO: Consolidate pattern with other patterns

    for timestamps_path in pl.Path(root_dir).glob(gaze_timestamp_pattern):
        raw_path = find_raw_path(timestamps_path)
        timestamps = load_timestamps_data(timestamps_path)
        raw_data = load_raw_data(raw_path)
        if len(raw_data) != len(timestamps):
            logger.warning(
                f"There is a mismatch between the number of raw data ({len(raw_data)}) "
                f"and the number of timestamps ({len(timestamps)})!"
            )
            size = min(len(raw_data), len(timestamps))
            raw_data = raw_data[:size]
            timestamps = timestamps[:size]
        yield from zip(raw_data, timestamps)
