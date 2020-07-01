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

from storage import SingleFileStorage
from gaze_producer import model
from observable import Observable

logger = logging.getLogger(__name__)


class ReferenceLocationStorage(SingleFileStorage, Observable):
    def __init__(self, rec_dir):
        super().__init__(rec_dir)
        self._reference_locations = {}
        self._load_from_disk()

    def add(self, reference_location):
        self._reference_locations[reference_location.frame_index] = reference_location

    def get_or_none(self, frame_index):
        return self._reference_locations.get(frame_index, None)

    def get_next(self, frame_index):
        found_index = min(
            idx for idx in self._reference_locations.keys() if idx > frame_index
        )
        return self._reference_locations[found_index]

    def get_previous(self, frame_index):
        found_index = max(
            idx for idx in self._reference_locations.keys() if idx < frame_index
        )
        return self._reference_locations[found_index]

    def get_in_range(self, frame_index_range):
        def in_range(ref):
            return frame_index_range[0] <= ref.frame_index <= frame_index_range[1]

        return [ref for ref in self._reference_locations.values() if in_range(ref)]

    def delete(self, reference_location):
        del self._reference_locations[reference_location.frame_index]

    def delete_all(self):
        self._reference_locations.clear()

    @property
    def _storage_file_name(self):
        return "reference_locations.msgpack"

    @property
    def _item_class(self):
        return model.ReferenceLocation

    @property
    def items(self):
        return self._reference_locations.values()
