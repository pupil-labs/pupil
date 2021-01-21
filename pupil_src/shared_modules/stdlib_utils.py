"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import typing
import functools
import itertools
import operator
import collections
import platform


def is_apple_silicone_platform() -> bool:
    if "Darwin" != platform.system():
        return False

    # Ex: Darwin Kernel Version 20.2.0: Wed Dec  2 20:40:21 PST 2020; root:xnu-7195.60.75~1/RELEASE_ARM64_T810
    # Ex: Darwin Kernel Version 19.6.0: Tue Nov 10 00:10:30 PST 2020; root:xnu-6153.141.10~1/RELEASE_X86_64
    os_version = os.uname().version
    return "ARM64" in os_version


is_none = functools.partial(operator.is_, None)


is_not_none = functools.partial(operator.is_not, None)


class sliceable_deque(collections.deque):
    """
    deque subclass with support for slicing.
    """

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(
                itertools.islice(self, index.start, index.stop, index.step),
                maxlen=self.maxlen,
            )
        return collections.deque.__getitem__(self, index)


Unique_Element = collections.abc.Hashable
Unique_Key = collections.abc.Hashable
Unique_Key_Getter = typing.Callable[[Unique_Element], Unique_Key]
Unique_Select = typing.Callable[[Unique_Key, Unique_Key], Unique_Key]


class unique(collections.abc.Iterable):
    def __init__(
        self,
        it: typing.Iterable,
        key: Unique_Key_Getter = ...,
        select: Unique_Select = ...,
    ):
        self._it = list(it)
        self._key = key if key is not ... else lambda elem: elem
        self._select = select if select is not ... else lambda x, y: x

    def __iter__(self):
        by_key = collections.OrderedDict()

        for new_elem in self._it:
            key = self._key(new_elem)

            try:
                old_elem = by_key[key]
            except KeyError:
                old_elem = None

            if old_elem is None:
                by_key[key] = new_elem
            else:
                by_key[key] = self._select(old_elem, new_elem)

        return iter(by_key.values())
