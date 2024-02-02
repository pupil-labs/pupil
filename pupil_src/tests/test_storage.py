import os
import pytest

from storage import StorageItem, Storage, SingleFileStorage
from observable import Observable


def test_storage_item_public_api():
    uid1 = StorageItem.create_new_unique_id()
    uid2 = StorageItem.create_new_unique_id()

    assert isinstance(uid1, str), "StorageItem.create_new_unique_id must return an instance of str"
    assert len(uid1) > 0, "StorageItem.create_new_unique_id must return a non-empty string"
    assert uid1 != uid2, "StorageItem.create_new_unique_id must return a unique string"

    uid1 = StorageItem.create_unique_id_from_string("foo")
    uid2 = StorageItem.create_unique_id_from_string("bar")
    uid3 = StorageItem.create_unique_id_from_string("foo")

    assert isinstance(uid1, str), "StorageItem.create_unique_id_from_string must return an instance of str"
    assert len(uid1) > 0, "StorageItem.create_unique_id_from_string must return a non-empty string"
    assert uid1 != uid2, "StorageItem.create_unique_id_from_string must return a unique string in different input"
    assert uid1 == uid3, "StorageItem.create_unique_id_from_string must return the same string on same input"


def test_single_file_storage_public_api(tmpdir):
    plugin = DummyPlugin()
    storage = DummySingleFileStorage(rec_dir=tmpdir, plugin=plugin)

    item1 = DummyStorageItem(foo=999, bar=["abc", 3])
    item2 = DummyStorageItem(foo=None, bar=item1.as_tuple)

    storage.add(item1)
    storage.add(item2)
    assert len(storage.items) == 2, "Storage must contain the added items"

    file_path = os.path.join(tmpdir, "offline_data", storage._storage_file_name)

    assert not os.path.exists(file_path), "The storage file must not exist in the temporary directory"
    storage.save_to_disk()
    assert os.path.exists(file_path), "The storage file must exists after saving to disk"

    # Reset storage
    storage = DummySingleFileStorage(rec_dir=tmpdir, plugin=plugin)
    assert len(storage.items) == 0, "Storage must be empty on initialization"

    storage._load_from_disk()
    assert len(storage.items) == 2, "Storage must load from disk all the saved items"

    deserialized1, deserialized2 = storage.items
    deserialized2.bar = DummyStorageItem.from_tuple(deserialized2.bar)

    assert deserialized1.foo == deserialized2.bar.foo == 999, "Deserialization must yield the same values as the original"
    assert deserialized1.bar == deserialized2.bar.bar == ["abc", 3], "Deserialization must yield the same values as the original"
    assert deserialized2.foo == None, "Deserialization must yield the same values as the original"


##################################################


class DummyPlugin(Observable):
    def cleanup(self):
        pass


class DummyStorageItem(StorageItem):
    version = 123

    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

    @staticmethod
    def from_tuple(tuple_):
        foo, bar = tuple_
        return DummyStorageItem(foo=foo, bar=bar)

    @property
    def as_tuple(self):
        return (self.foo, self.bar)


class DummySingleFileStorage(SingleFileStorage):
    @property
    def _storage_file_name(self):
        return "dummy_items.xyz"

    def add(self, item):
        self.items.append(item)

    def delete(self, item):
        self.items.remove(item)

    @property
    def items(self):
        try:
            return self.__item_storage
        except AttributeError:
            self.__item_storage = []
            return self.__item_storage

    @property
    def _item_class(self):
        return DummyStorageItem
