"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import uuid
import pytest

from stdlib_utils import unique, is_none, is_not_none, lazy_property


def test_unique():
    arr = [1, 2, 3, 1, 4, 2, 2, 4, 5, 6, 0]
    assert list(unique(arr)) == [1, 2, 3, 4, 5, 6, 0]
    assert list(unique(arr)) == list(unique(unique(arr)))

    assert list(unique(arr, key=lambda _: 0)) == [1]
    assert list(unique(arr, key=lambda _: 0, select=lambda old, new: new)) == [0]
    assert list(unique(arr, select=lambda old, new: old*10 + new)) == [11, 222, 3, 44, 5, 6, 0]


def test_operators():

    assert is_none(None) == True
    assert is_not_none(None) == False

    things = [0, 5, [], [1,2,3], "", "abc"]

    for thing in things:
        assert is_none(thing) == False
        assert is_not_none(thing) == True


class FooWithLazyBar:
    def __init__(self, value):
        self.bar_lazy_init_call_count = 0
        self.bar_lazy_init_value = value

    def _bar_init_fn(self):
        self.bar_lazy_init_call_count += 1
        return self.bar_lazy_init_value

# Can't define these inside the class definition
FooWithLazyBar.bar = lazy_property(FooWithLazyBar._bar_init_fn)
FooWithLazyBar.readonly_bar = lazy_property(FooWithLazyBar._bar_init_fn, readonly=True)


def test_lazy_property():
    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()

    uut = FooWithLazyBar(uuid1)

    assert uut.bar_lazy_init_call_count == 0, "The lazy init closure shouldn't run before the first access"
    assert uut.bar == uuid1, "The lazily initialized value should be equal to the return of the closure"
    assert uut.bar_lazy_init_call_count == 1, "The lazy init closure should run on first access"
    assert uut.bar == uuid1, "The lazily initialized value should be equal to the return of the closure"
    assert uut.bar_lazy_init_call_count == 1, "The lazy init closure should only run once"

    uut = FooWithLazyBar(uuid1)
    uut.bar = uuid2

    assert uut.bar == uuid2, "The lazily initialized value should be equal to the latest value it was assigned"
    assert uut.bar_lazy_init_call_count == 0, "The lazy init closure shouldn't run at all if the first access happened after an assignment"

    assert uut.readonly_bar == uuid1, "The readonly lazily initialized value should be equal to the return of the closure"

    with pytest.raises(AttributeError):
        uut.readonly_bar = uuid2
        assert False, "The readonly lazily initialized property should not be settable"
