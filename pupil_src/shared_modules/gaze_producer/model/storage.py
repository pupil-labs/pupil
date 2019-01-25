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
import random

import file_methods as fm

logger = logging.getLogger(__name__)


class StorageItem(abc.ABC):
    @property
    @abc.abstractmethod
    def version(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_tuple(tuple_):
        pass

    @property
    @abc.abstractmethod
    def as_tuple(self):
        pass

    @staticmethod
    def create_new_unique_id():
        # will give a string like e.g. "04bfd332"
        return "{:0>8x}".format(random.getrandbits(32))


class Storage(abc.ABC):
    def __init__(self, plugin):
        plugin.add_observer("cleanup", self._on_cleanup)

    def __iter__(self):
        return iter(self.items)

    @abc.abstractmethod
    def add(self, item):
        pass

    @abc.abstractmethod
    def delete(self, item):
        pass

    @property
    @abc.abstractmethod
    def items(self):
        pass

    @property
    @abc.abstractmethod
    def _item_class(self):
        pass

    @abc.abstractmethod
    def save_to_disk(self):
        pass

    @abc.abstractmethod
    def _load_from_disk(self):
        pass

    def _save_data_to_file(self, filename, data):
        dict_representation = {"version": self._item_class.version, "data": data}
        fm.save_object(dict_representation, filename)

    def _load_data_from_file(self, filename):
        try:
            dict_representation = fm.load_object(filename)
        except FileNotFoundError:
            return None
        if dict_representation.get("version", None) != self._item_class.version:
            logger.warning(
                "Data in {} is in old file format. Will not load these!".format(
                    filename
                )
            )
            return None
        return dict_representation.get("data", None)

    def _on_cleanup(self):
        self.save_to_disk()
