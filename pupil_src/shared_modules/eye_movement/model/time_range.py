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


class Time_Range:
    def __init__(self, start_time: float, end_time: float):
        assert start_time <= end_time
        self.start_time = start_time
        self.end_time = end_time

    def contains(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time

    def intersection(self, other: "Time_Range") -> t.Optional["Time_Range"]:
        start_time = max(self.start_time, other.start_time)
        end_time = min(self.end_time, other.end_time)
        if start_time <= end_time:
            return Time_Range(start_time=start_time, end_time=end_time)
        else:
            return None

    def union(self, other: "Time_Range") -> "Time_Range":
        """Warning: This method doesn't guarantee that the union is continuous."""
        start_time = min(self.start_time, other.start_time)
        end_time = max(self.end_time, other.end_time)
        return Time_Range(start_time=start_time, end_time=end_time)

    def countinuous_union(self, other: "Time_Range") -> t.Optional["Time_Range"]:
        if not self.intersection(other):
            return None
        return self.union(other)
