"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as T
from collections import deque

import numpy as np


class RealtimeMatcher:
    def __init__(self):
        self.min_pupil_confidence = 0.6
        self._caches = (deque(), deque())
        self.recently_estimated_framerate = 1 / 120
        self.framerate_estimation_smoothing_factor = 1 / 50
        self.sample_cutoff = 10

    def is_cache_valid(self, cache):
        return len(cache) >= 2

    def estimate_frame_rate_raw(self, cache):
        return np.mean(np.diff([d["timestamp"] for d in cache]))

    def estimate_framerate_smoothed(self, eye0_cache, eye1_cache):
        if self.is_cache_valid(eye0_cache) and self.is_cache_valid(eye1_cache):
            eye0_framerate_raw = self.estimate_frame_rate_raw(eye0_cache)
            eye1_framerate_raw = self.estimate_frame_rate_raw(eye1_cache)
            estimated_framerate_raw = max(eye0_framerate_raw, eye1_framerate_raw)
        elif self.is_cache_valid(eye0_cache):
            estimated_framerate_raw = self.estimate_frame_rate_raw(eye0_cache)
        elif self.is_cache_valid(eye1_cache):
            estimated_framerate_raw = self.estimate_frame_rate_raw(eye1_cache)
        else:
            return self.recently_estimated_framerate

        self.recently_estimated_framerate += (
            estimated_framerate_raw - self.recently_estimated_framerate
        ) * self.framerate_estimation_smoothing_factor
        return self.recently_estimated_framerate

    def map_batch(self, pupil_list):
        current_caches = self._caches
        self._caches = (deque(), deque())
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))

        self._caches = current_caches
        return results

    def on_pupil_datum(self, p) -> T.Iterator:
        """Returns a list with either zero, one or two pupil datums.
        - zero: not enough data in queue
        - one: no binocular match possible
        - two: binocular match
        """
        self._caches[p["id"]].append(p)
        temporal_cutoff = 2 * self.estimate_framerate_smoothed(*self._caches)

        # map low confidence pupil data monocularly
        if (
            self._caches[0]
            and self._caches[0][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[0].popleft()
            yield [p]
        elif (
            self._caches[1]
            and self._caches[1][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[1].popleft()
            yield [p]
        # map high confidence data binocularly if available
        elif self._caches[0] and self._caches[1]:
            # we have binocular data
            if self._caches[0][0]["timestamp"] < self._caches[1][0]["timestamp"]:
                p0 = self._caches[0].popleft()
                p1 = self._caches[1][0]
                older_pt = p0
            else:
                p0 = self._caches[0][0]
                p1 = self._caches[1].popleft()
                older_pt = p1

            if abs(p0["timestamp"] - p1["timestamp"]) < temporal_cutoff:
                yield [p0, p1]
            else:
                yield [older_pt]

        elif len(self._caches[0]) > self.sample_cutoff:
            p = self._caches[0].popleft()
            yield [p]
        elif len(self._caches[1]) > self.sample_cutoff:
            p = self._caches[1].popleft()
            yield [p]
