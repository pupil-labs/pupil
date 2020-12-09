"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import collections.abc
import copy
import logging
import os
import pickle
import traceback as tb
import types
from glob import iglob
from pathlib import Path

import msgpack
import numpy as np


assert (
    msgpack.version[0] == 1
), "msgpack out of date, please upgrade to version (1, 0, 0)"


logger = logging.getLogger(__name__)
UnpicklingError = pickle.UnpicklingError

PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])


class Persistent_Dict(dict):
    """a dict class that uses pickle to save inself to file"""

    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = os.path.expanduser(file_path)
        try:
            self.update(**load_object(self.file_path, allow_legacy=False))
        except IOError:
            logger.debug(
                f"Session settings file '{self.file_path}' not found."
                " Will make new one on exit."
            )
        except Exception:  # KeyError, EOFError
            logger.warning(
                f"Session settings file '{self.file_path}'could not be read."
                " Will overwrite on exit."
            )
            logger.debug(tb.format_exc())

    def save(self):
        d = {}
        d.update(self)
        save_object(d, self.file_path)

    def close(self):
        self.save()


def _load_object_legacy(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "rb") as fh:
        data = pickle.load(fh, encoding="bytes")
    return data


def load_object(file_path, allow_legacy=True):
    import gc

    file_path = Path(file_path).expanduser()
    with file_path.open("rb") as fh:
        try:
            gc.disable()  # speeds deserialization up.
            data = msgpack.unpack(fh, strict_map_key=False)
        except Exception as e:
            if not allow_legacy:
                raise e
            else:
                logger.info(
                    "{} has a deprecated format: Will be updated on save".format(
                        file_path
                    )
                )
                data = _load_object_legacy(file_path)
        finally:
            gc.enable()
    return data


def save_object(object_, file_path):
    def ndarrray_to_list(
        o, _warned=[False]
    ):  # Use a mutlable default arg to hold a fn interal temp var.
        if isinstance(o, np.ndarray):
            if not _warned[0]:
                logger.warning(
                    "numpy array will be serialized as list. Invoked at:\n"
                    + "".join(tb.format_stack())
                )
                _warned[0] = True
            return o.tolist()
        return o

    file_path = Path(file_path).expanduser()
    with file_path.open("wb") as fh:
        msgpack.pack(object_, fh, use_bin_type=True, default=ndarrray_to_list)


class Incremental_Legacy_Pupil_Data_Loader(object):
    def __init__(self, directory=""):
        self.file_loc = os.path.join(directory, "pupil_data")

    def __enter__(self):
        self.file_handle = open(self.file_loc, "rb")
        self.unpacker = msgpack.Unpacker(
            self.file_handle, use_list=False, strict_map_key=False
        )
        self.num_key_value_pairs = self.unpacker.read_map_header()
        self._skipped = True
        return self

    def __exit__(self, *exc):
        self.file_handle.close()

    def topic_values_pairs(self):
        for _ in range(self.num_key_value_pairs):
            yield self.unpacker.unpack(), self._next_values()

    def _next_values(self):
        for _ in range(self.unpacker.read_array_header()):
            yield self.unpacker.unpack()


def load_pldata_file(directory, topic):
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    msgpack_file = os.path.join(directory, topic + ".pldata")
    try:
        data = collections.deque()
        topics = collections.deque()
        data_ts = np.load(ts_file)
        with open(msgpack_file, "rb") as fh:
            for topic, payload in msgpack.Unpacker(
                fh, use_list=False, strict_map_key=False
            ):
                data.append(Serialized_Dict(msgpack_bytes=payload))
                topics.append(topic)
    except FileNotFoundError:
        data = []
        data_ts = []
        topics = []

    return PLData(data, data_ts, topics)


class PLData_Writer(object):
    """docstring for PLData_Writer"""

    def __init__(self, directory, name):
        super().__init__()
        self.directory = directory
        self.name = name
        self.ts_queue = collections.deque()
        file_name = name + ".pldata"
        self.file_handle = open(os.path.join(directory, file_name), "wb")

    def append(self, datum):
        datum_serialized = msgpack.packb(datum, use_bin_type=True)
        self.append_serialized(datum["timestamp"], datum["topic"], datum_serialized)

    def append_serialized(self, timestamp, topic, datum_serialized):
        self.ts_queue.append(timestamp)
        pair = msgpack.packb((topic, datum_serialized), use_bin_type=True)
        self.file_handle.write(pair)

    def extend(self, data):
        for datum in data:
            self.append(datum)

    def close(self):
        self.file_handle.close()
        self.file_handle = None

        ts_file = self.name + "_timestamps.npy"
        ts_path = os.path.join(self.directory, ts_file)
        np.save(ts_path, self.ts_queue)
        self.ts_queue = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def next_export_sub_dir(root_export_dir):
    # match any sub directories or files a three digit pattern
    pattern = os.path.join(root_export_dir, "[0-9][0-9][0-9]")
    existing_subs = sorted(iglob(pattern))
    try:
        latest = os.path.split(existing_subs[-1])[-1]
        next_sub_dir = "{:03d}".format(int(latest) + 1)
    except IndexError:
        next_sub_dir = "000"

    return os.path.join(root_export_dir, next_sub_dir)


class _Empty(object):
    def purge_cache(self):
        pass


class Serialized_Dict(object):
    __slots__ = ["_ser_data", "_data"]
    cache_len = 100
    _cache_ref = [_Empty()] * cache_len
    MSGPACK_EXT_CODE = 13

    def __init__(self, python_dict=None, msgpack_bytes=None):
        if type(python_dict) is dict:
            self._ser_data = msgpack.packb(
                python_dict, use_bin_type=True, default=self.packing_hook
            )
        elif type(msgpack_bytes) is bytes:
            self._ser_data = msgpack_bytes
        else:
            raise ValueError(
                "You did not supply mapping or payload to Serialized_Dict."
            )
        self._data = None

    def _deser(self):
        if not self._data:
            self._data = msgpack.unpackb(
                self._ser_data,
                use_list=False,
                object_hook=self.unpacking_object_hook,
                ext_hook=self.unpacking_ext_hook,
                strict_map_key=False,
            )
            self._cache_ref.pop(0).purge_cache()
            self._cache_ref.append(self)

    def __getstate__(self):
        return self._ser_data

    def __setstate__(self, msgpack_bytes):
        self._ser_data = msgpack_bytes
        self._data = None

    @classmethod
    def unpacking_object_hook(self, obj):
        if type(obj) is dict:
            return types.MappingProxyType(obj)

    @classmethod
    def packing_hook(self, obj):
        if isinstance(obj, self):
            return msgpack.ExtType(self.MSGPACK_EXT_CODE, obj.serialized)
        raise TypeError("can't serialize {}({})".format(type(obj), repr(obj)))

    @classmethod
    def unpacking_ext_hook(self, code, data):
        if code == self.MSGPACK_EXT_CODE:
            return self(msgpack_bytes=data)
        return msgpack.ExtType(code, data)

    def purge_cache(self):
        self._data = None

    @property
    def serialized(self):
        return self._ser_data

    def __setitem__(self, key, item):
        raise NotImplementedError()

    def __getitem__(self, key):
        self._deser()
        return self._data[key]

    def __repr__(self):
        self._deser()
        return "Serialized_Dict({})".format(repr(self._data))

    @property
    def len(self):
        """Replacement implementation for __len__

        If __len__ is defined numpy will recognize this as nested structure and
        start deserializing everything instead of using this object as it is.
        """
        self._deser()
        return len(self._data)

    def __delitem__(self, key):
        raise NotImplementedError()

    def get(self, key, default):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        raise NotImplementedError()

    def copy(self):
        self._deser()
        return self._data.copy()

    def __deepcopy__(self, memo=None):
        return _recursive_deep_copy(self)

    def has_key(self, k):
        self._deser()
        return k in self._data

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def keys(self):
        self._deser()
        return self._data.keys()

    def values(self):
        self._deser()
        return self._data.values()

    def items(self):
        self._deser()
        return self._data.items()

    def pop(self, *args):
        raise NotImplementedError()

    def __cmp__(self, dict_):
        self._deser()
        return self._data.__cmp__(dict_)

    def __contains__(self, item):
        self._deser()
        return item in self._data

    def __iter__(self):
        self._deser()
        return iter(self._data)

    def _deep_copy_serialized_dict(self):
        dict_copy = self._deep_copy_dict()
        return Serialized_Dict(python_dict=dict_copy)

    def _deep_copy_dict(self):
        def unpacking_ext_hook(self, code, data):
            if code == self.MSGPACK_EXT_CODE:
                return type(self)(msgpack_bytes=data)._deep_copy_dict()
            return msgpack.ExtType(code, data)

        return msgpack.unpackb(
            self._ser_data,
            use_list=False,
            ext_hook=unpacking_ext_hook,
        )


def _recursive_deep_copy(item):

    if isinstance(item, collections.abc.Mapping):
        _item_dict = {k: _recursive_deep_copy(v) for k, v in item.items()}
        if isinstance(item, types.MappingProxyType):
            return _item_dict
        else:
            return type(item)(_item_dict)

    if isinstance(item, collections.abc.Sequence) and not isinstance(item, str):
        return type(item)([_recursive_deep_copy(el) for el in item])

    return copy.deepcopy(item)


def bench_save():
    import time

    # in recorder
    start = time.time()
    data = []
    inters = 200 * 60 * 60  # 1h recording
    dummy_datum = {
        "topic": "pupil",
        "circle_3d": {
            "center": [0.0, -0.0, 0.0],
            "normal": [0.0, -0.0, 0.0],
            "radius": 0.0,
        },
        "confidence": 0.0,
        "timestamp": 0.9351908409998941,
        "diameter_3d": 0.0,
        "ellipse": {"center": [96.0, 96.0], "axes": [0.0, 0.0], "angle": 90.0},
        "norm_pos": [0.5, 0.5],
        "diameter": 0.0,
        "sphere": {
            "center": [-2.2063483765091934, 0.0836648916925231, 48.13110450930929],
            "radius": 12.0,
        },
        "projected_sphere": {
            "center": [67.57896110256269, 97.07772787219814],
            "axes": [309.15558975219403, 309.15558975219403],
            "angle": 90.0,
        },
        "model_confidence": 1.0,
        "model_id": 1,
        "model_birth_timestamp": 640.773183,
        "theta": 0,
        "phi": 0,
        "method": "3d c++",
        "id": 0,
    }

    with open("test", "wb") as fb:
        packer = msgpack.Packer(use_bin_type=True)
        for x in range(inters):
            a = "pupil", msgpack.packb(dummy_datum, use_bin_type=True)
            b = "pupil", msgpack.packb(dummy_datum, use_bin_type=True)
            c = "gaze", msgpack.packb(dummy_datum, use_bin_type=True)
            aa = "aa", msgpack.packb({"test": {"nested": True}}, use_bin_type=True)
            fb.write(packer.pack(a))
            fb.write(packer.pack(b))
            fb.write(packer.pack(c))
            fb.write(packer.pack(aa))
    print("generated and saved in %s" % (time.time() - start))


def bench_load():

    import time

    start = time.time()
    pupil_data = load_pupil_data_file("test")
    print(pupil_data.keys())
    print("loaded in %s" % (time.time() - start))


if __name__ == "__main__":
    import sys

    # d = load_object("/Users/mkassner/Downloads/000/pupil_data")["gaze_positions"]
    # size = len(d)
    # print(size)
    # # del d
    # l = []
    # for p in range(size):
    #     l.append(Serialized_Dict(d[p]))
    #     l[-1]['timestamp']
    # del(d)
    # print(size)
    # print("slfrf")
    # from time import sleep
    # sleep(10)
    bench_save()
    from time import sleep

    # sleep(3)
    bench_load()
