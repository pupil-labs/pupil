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
import collections
from .segment import Classified_Segment
from observable import Observable
from storage import SingleFileStorage
import player_methods as pm


class Classified_Segment_Storage(SingleFileStorage, Observable):
    def __init__(self, plugin, rec_dir):
        super().__init__(plugin=plugin, rec_dir=rec_dir)
        self._queue = collections.deque()
        self._affiliator = pm.Affiliator()
        self._load_from_disk()

        g_pool = plugin.g_pool
        self._get_global_timestamps = lambda: g_pool.timestamps

    def add(self, item):
        self._queue.append(item)

    def delete(self, item):
        self._queue.remove(item)

    @property
    def items(self):
        return self._queue

    @property
    def _item_class(self):
        return Classified_Segment

    @property
    def _storage_file_name(self):
        return "eye_movement.msgpack"

    def _load_from_disk(self):
        super()._load_from_disk()
        self.finalize()

    @property
    def _timestamps(self) -> t.Iterable[float]:
        return self._get_global_timestamps()

    def __len__(self):
        return len(self._affiliator)

    def __getitem__(self, index: int) -> Classified_Segment:
        return self._affiliator[index]

    def index(self, item: Classified_Segment) -> t.Optional[int]:
        return self._affiliator.data.index(item)

    def clear(self):
        self._queue.clear()
        self._affiliator = pm.Affiliator()

    def finalize(self):
        start_timestamps = [s.start_frame_timestamp for s in self._queue]
        end_timestamps = [s.end_frame_timestamp for s in self._queue]
        self._affiliator = pm.Affiliator(self._queue, start_timestamps, end_timestamps)

    def segments_in_frame(self, frame) -> t.Iterable[Classified_Segment]:
        frame_window = pm.enclosing_window(self._timestamps, frame.index)
        return self.segments_in_timestamp_window(frame_window)

    def segments_in_range(self, range) -> t.Iterable[Classified_Segment]:
        range_window = pm.exact_window(self._timestamps, range)
        return self.segments_in_timestamp_window(range_window)

    def segments_in_timestamp_window(
        self, timestamp_window
    ) -> t.Iterable[Classified_Segment]:
        if len(self._affiliator) <= 0:
            return []
        return self._affiliator.by_ts_window(timestamp_window)
