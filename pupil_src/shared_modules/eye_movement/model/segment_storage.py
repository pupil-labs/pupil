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
from eye_movement.utils import MsgPack_Serialized_Segment
import file_methods as fm


class Classified_Segment_Storage(abc.ABC):

    def __getitem__(self, key):
        ...

    def get(self, key, default):
        ...

    def to_dict(self) -> dict:
        ...

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        ...

    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        ...


class Classified_Segment_Dict_Storage(Classified_Segment_Storage):

    def __init__(self, python_dict: dict):
        self._python_dict = python_dict

    def __getitem__(self, key):
        return self._python_dict.__getitem__(key)

    def get(self, key, default):
        return self._python_dict.get(key, default)

    def to_dict(self) -> dict:
        return self._python_dict

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return fm.Serialized_Dict(python_dict=self._python_dict)

    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        serialized_dict = fm.Serialized_Dict(python_dict=self._python_dict)
        return serialized_dict.serialized


class Classified_Segment_Serialized_Dict_Storage(Classified_Segment_Storage):

    def __init__(self, serialized_dict: fm.Serialized_Dict):
        self._serialized_dict = serialized_dict

    def __getitem__(self, key):
        return self._serialized_dict.__getitem__(key)

    def get(self, key, default):
        return self._serialized_dict.get(key, default)

    def to_dict(self) -> dict:
        return dict(self._serialized_dict)

    def to_serialized_dict(self) -> fm.Serialized_Dict:
        return self._serialized_dict

    def to_msgpack(self) -> MsgPack_Serialized_Segment:
        serialized_dict = self._serialized_dict
        return serialized_dict.serialized


class Classified_Segment_MsgPack_Storage(Classified_Segment_Serialized_Dict_Storage):

    def __init__(self, msgpack_bytes: MsgPack_Serialized_Segment):
        serialized_dict = fm.Serialized_Dict(msgpack_bytes=msgpack_bytes)
        super().__init__(serialized_dict=serialized_dict)
