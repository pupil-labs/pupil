"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import os

from gaze_producer import model

logger = logging.getLogger(__name__)


class SingleFileStorage(model.storage.Storage, abc.ABC):
    """
    Storage that can save and load all items from / to a single file
    """
    def __init__(self, rec_dir, plugin):
        super().__init__(plugin)
        self._rec_dir = rec_dir

    def save_to_disk(self):
        item_tuple_list = [item.as_tuple for item in self.items]
        self._save_data_to_file(self._storage_file_path, item_tuple_list)

    def _load_from_disk(self):
        item_tuple_list = self._load_data_from_file(self._storage_file_path)
        if item_tuple_list:
            for item_tuple in item_tuple_list:
                item = self._item_class.from_tuple(item_tuple)
                self.add(item)

    @property
    @abc.abstractmethod
    def _storage_file_name(self):
        pass

    @property
    def _storage_file_path(self):
        return os.path.join(self._storage_folder_path, self._storage_file_name)

    @property
    def _storage_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")
