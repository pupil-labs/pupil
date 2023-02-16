"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import pytest
from stdlib_utils import is_none, is_not_none, unique


def test_unique():
    arr = [1, 2, 3, 1, 4, 2, 2, 4, 5, 6, 0]
    assert list(unique(arr)) == [1, 2, 3, 4, 5, 6, 0]
    assert list(unique(arr)) == list(unique(unique(arr)))

    assert list(unique(arr, key=lambda _: 0)) == [1]
    assert list(unique(arr, key=lambda _: 0, select=lambda old, new: new)) == [0]
    assert list(unique(arr, select=lambda old, new: old * 10 + new)) == [
        11,
        222,
        3,
        44,
        5,
        6,
        0,
    ]


def test_operators():
    assert is_none(None) == True
    assert is_not_none(None) == False

    things = [0, 5, [], [1, 2, 3], "", "abc"]

    for thing in things:
        assert is_none(thing) == False
        assert is_not_none(thing) == True
