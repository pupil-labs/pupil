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
import glob
import fnmatch
import re
import os
import cv2
import numpy as np
import av

from typing import Sequence, Iterator
from camera_models import load_intrinsics


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
        self._is_valid = self.check_valid()

    @property
    def is_valid(self):
        return self._is_valid

    def check_valid(self):
        try:
            cont = av.open(self.path)
            n = cont.decode(video=0)
            _ = next(n)
        except av.AVError:
            return False
        else:
            cont.seek(0)
            self.cont = cont
            return True

    def load_valid_container(self):
        if self.is_valid:
            return self.cont

    def load_ts(self):
        self.ts = np.load(self.ts_loc)

    @property
    def name(self) -> str:
        file_ = os.path.split(self.path)[1]
        return os.path.splitext(file_)[0]

    @property
    def ts_loc(self) -> str:
        return os.path.join(self.base, f"{self.name}_timestamps.npy")

    @property
    def base(self) -> str:
        return os.path.split(self.path)[0]

    @property
    def timestamps(self) -> np.ndarray:
        if self.ts is None:
            self.load_ts()
        return self.ts


class VideoSet:
    def __init__(self, rec: str, name: str, fill_gaps: bool):
        self.rec = rec
        self.name = name
        self.fill_gaps = fill_gaps
        self.video_exts = VIDEO_EXTS
        self._videos = sorted(self.fetch_videos(), key=lambda v: v.path)
        self._containers = [vid.load_valid_container() for vid in self.videos]

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
        yield from (
            Video(loc)
            for ext in self.video_exts
            for loc in glob.iglob(os.path.join(self.rec, f"{self.name}*.{ext}"))
            if (self.fill_gaps or Video(loc).is_valid)
        )

    def build_lookup(self):
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
        lookup = self._setup_lookup(loaded_ts)
        for container_idx, vid in enumerate(self.videos):
            mask = np.isin(lookup.timestamp, vid.timestamps)
            lookup.container_frame_idx[mask] = np.arange(vid.timestamps.size)
            if vid.is_valid:
                lookup.container_idx[mask] = container_idx
        self.lookup = lookup

    def save_lookup(self):
        np.save(self.lookup_loc, self.lookup)

    def load_lookup(self):
        self.lookup = np.load(self.lookup_loc).view(np.recarray)

    def load_or_build_lookup(self):
        try:
            self.load_lookup()
        except FileNotFoundError:
            self.build_lookup()
            self.save_lookup()
        if not self.fill_gaps:
            self._remove_filled_gaps()

    def _loaded_ts_sorted(self) -> np.ndarray:
        loaded_ts = [vid.timestamps for vid in self.videos]
        all_ts = np.concatenate(loaded_ts)
        return all_ts

    def _remove_filled_gaps(self):
        cont_idc = self.lookup.container_idx
        self.lookup = self.lookup[cont_idc > -1]

    def _fill_gaps(self, timestamps: np.ndarray) -> np.ndarray:
        time_diff = np.diff(timestamps)
        median_time_diff = np.median(time_diff)
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
            ]
        )
        lookup = np.empty(timestamps.size, dtype=lookup_entry).view(np.recarray)
        lookup.timestamp = timestamps
        lookup.container_idx = -1  # virtual container by default
        return lookup


class RenameSet:
    class RenameFile:
        def __init__(self, path):
            self.path = path

        @property
        def name(self):
            file_name = os.path.split(self.path)[1]
            return os.path.splitext(file_name)[0]

        def rename(self, source_pattern, destination_name):
            # source_pattern: Pupil Cam(0|1) ID0
            # destination_name: eye0
            if re.match(source_pattern, self.name):
                new_path = re.sub(source_pattern, destination_name, self.path)
                logger.info(
                    'Renaming "{}" to "{}"'.format(
                        self.name, os.path.split(new_path)[1]
                    )
                )
                try:
                    os.rename(self.path, new_path)
                except FileExistsError:
                    # Only happens on Windows. Behavior on Unix is to
                    # overwrite the existing file.To mirror this behaviour
                    # we need to delete the old file and try renaming the
                    # new one again.
                    os.remove(self.path)
                    os.rename(self.path, new_path)

        def rewrite_time(self, destination_name):
            timestamps = np.fromfile(self.path, dtype=">f8")
            logger.info('Creating "{}_timestamps.npy"'.format(self.name))
            timestamp_loc = os.path.join(
                os.path.dirname(self.path), "{}_timestamps.npy".format(self.name)
            )
            np.save(timestamp_loc, timestamps)

    def __init__(self, rec_dir, pattern, exts=VIDEO_TIME_EXTS):
        self.rec_dir = rec_dir
        self.pattern = os.path.join(rec_dir, pattern)
        self.existsting_files = self.get_existsting_files(self.pattern, exts)

    def rename(self, source_pattern, destination_name):
        for r in self.existsting_files:
            self.RenameFile(r).rename(source_pattern, destination_name)

    def rewrite_time(self, destination_name):
        for r in self.existsting_files:
            self.RenameFile(r).rewrite_time(destination_name)

    def load_intrinsics(self):
        def _load_intrinsics(file_name, name):
            try:
                video = av.open(file_name, "r")
            except av.AVError:
                frame_size = (480, 360)
            else:
                frame_size = (
                    video.streams.video[0].format.width,
                    video.streams.video[0].format.height,
                )
                del video
            intrinsics = load_intrinsics(self.rec_dir, name, frame_size)
            intrinsics.save(self.rec_dir, "world")

        for fn in self.existsting_files:
            if fnmatch.fnmatch(fn, "*Pupil Cam1 ID2*"):
                _load_intrinsics(fn, "Pupil Cam1 ID2")
            elif fnmatch.fnmatch(fn, "*Logitech Webcam C930e*"):
                _load_intrinsics(fn, "Logitech Webcam C930e")

    def get_existsting_files(self, pattern, exts):
        existsting_files = []
        for loc in glob.glob(pattern):
            file_name = os.path.split(loc)[1]
            name = os.path.splitext(file_name)[0]
            potential_locs = [
                os.path.join(self.rec_dir, name + "." + ext) for ext in exts
            ]
            existsting_files.extend(
                [loc for loc in potential_locs if os.path.exists(loc)]
            )
        return existsting_files
