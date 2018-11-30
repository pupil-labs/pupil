"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os

import file_methods as fm
from gaze_producer.model.reference_location import ReferenceLocation
from observable import Observable

logger = logging.getLogger(__name__)


class ReferenceLocationStorage(Observable):
    def __init__(self, rec_dir, plugin):
        self._rec_dir = rec_dir
        self._reference_locations = {}
        self.load_from_disk()
        plugin.add_observer("cleanup", self._on_cleanup)

    def add(self, screen_pos, frame_index, timestamp):
        reference_location = ReferenceLocation(screen_pos, frame_index, timestamp)
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

    def delete(self, reference_location):
        del self._reference_locations[reference_location.frame_index]

    def delete_all(self):
        self._reference_locations = {}

    def __iter__(self):
        return iter(self._reference_locations.values())

    def _on_cleanup(self):
        self.save_to_disk()

    def save_to_disk(self):
        data = [
            (ref.timestamp, ref.frame_index, ref.screen_pos)
            for ref in self._reference_locations.values()
        ]
        dict_representation = {"version": ReferenceLocation.version, "data": data}
        fm.save_object(dict_representation, self._reference_locations_file)

    def load_from_disk(self):
        try:
            dict_representation = fm.load_object(self._reference_locations_file)
        except FileNotFoundError:
            return
        if dict_representation.get("version", None) != ReferenceLocation.version:
            logger.warning(
                "Found reference locations in old file format. Will not load these!"
            )
            return
        for datum in dict_representation["data"]:
            timestamp, frame_index, screen_pos = datum
            self.add(screen_pos, frame_index, timestamp)

    @property
    def _reference_locations_file(self):
        return os.path.join(self._rec_dir, "offline_data", "reference_locations")
