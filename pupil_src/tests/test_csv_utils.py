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
from csv_utils import read_key_value_file, write_key_value_file


@pytest.fixture
def testfile(tmpdir):
    return tmpdir.join("test.csv")


def test_read_write_key_value_file(testfile):
    test = {"foo": "bar", "oh": 'rl"abc"y', "it was": "not 🚨"}
    test_append = {"jo": "ho"}
    test_updated = test.copy()
    test_updated.update(test_append)

    # Test write+read
    with open(testfile, "w", encoding="utf-8", newline="") as csvfile:
        write_key_value_file(csvfile, test)
    with open(testfile, encoding="utf-8", newline="") as csvfile:
        result = read_key_value_file(csvfile)
    assert test == result, (test, result)

    # Test write+append (same keys)+read
    with open(testfile, "w", encoding="utf-8", newline="") as csvfile:
        write_key_value_file(csvfile, test)
        write_key_value_file(csvfile, test, append=True)
    with open(testfile, encoding="utf-8", newline="") as csvfile:
        result = read_key_value_file(csvfile)
    assert test == result

    # Test write+append (different keys)+read
    with open(testfile, "w", encoding="utf-8", newline="") as csvfile:
        write_key_value_file(csvfile, test)
        write_key_value_file(csvfile, test_append, append=True)
    with open(testfile, encoding="utf-8", newline="") as csvfile:
        result = read_key_value_file(csvfile)
    assert test_updated == result
