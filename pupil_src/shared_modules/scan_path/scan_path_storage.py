"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

from storage import StorageItem, SingleFileStorage
from observable import Observable


logger = logging.getLogger(__name__)


class ScanPathItem(StorageItem):
    version = 1

    @staticmethod
    def from_tuple(tuple_):
        return ScanPathItem(*tuple_)

    @property
    def as_tuple(self):
        return (
            self._frame_index,
            self._gaze_datums,
        )

    @property
    def index(self):
        return self._frame_index

    @property
    def data(self):
        return self._gaze_datums

    def __init__(self, frame_index, gaze_datums):
        super().__init__()
        self._frame_index = frame_index
        self._gaze_datums = gaze_datums


class ScanPathStorage(SingleFileStorage, Observable):

    def __init__(self, rec_dir, plugin):
        super().__init__(rec_dir, plugin)
        self._cache = {}
        self._load_from_disk()

    @property
    def is_completed(self) -> bool:
        return False #FIXME

    def mark_completed(self):
        pass #FIXME

    def get(self, frame_index):
        try:
            return self._cache[frame_index].data
        except KeyError:
            return None

    def clear(self):
        self._cache = {}

    # SingleFileStorage

    @property
    def _storage_file_name(self):
        return "scan_path_cache.msgpack"

    def add(self, item):
        self._cache[item.index] = item

    def delete(self, item):
        del self._cache[item.index]

    @property
    def items(self):
        return sorted(self._cache.values(), key=lambda item: item.index)

    @property
    def _item_class(self):
        return ScanPathItem
