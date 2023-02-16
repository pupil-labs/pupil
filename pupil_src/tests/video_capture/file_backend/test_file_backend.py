"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
from multiprocessing import cpu_count
from types import SimpleNamespace

import av
import pytest
from video_capture.base_backend import NoMoreVideoError
from video_capture.file_backend import Decoder, File_Source, OnDemandDecoder

from ..common import broken_data, multiple_data, single_data


@pytest.fixture
def single_fill_gaps():
    """Returns single data"""
    return File_Source(SimpleNamespace(), source_path=single_data, fill_gaps=True)


@pytest.fixture
def multiple_fill_gaps():
    """Returns multiple data"""
    return File_Source(SimpleNamespace(), source_path=multiple_data, fill_gaps=True)


@pytest.fixture
def broken_fill_gaps():
    """Returns broken data"""
    return File_Source(SimpleNamespace(), source_path=broken_data)


def test_file_source_recent_events():
    """
    recent_events setup correct or not
    """
    file_source = File_Source(
        SimpleNamespace(), source_path=single_data, timing="external"
    )
    assert file_source.recent_events == file_source.recent_events_external_timing
    file_source = File_Source(SimpleNamespace(), source_path=single_data, timing=None)
    assert file_source.recent_events == file_source.recent_events_own_timing


def test_file_source_init(single_fill_gaps):
    assert single_fill_gaps.initialised is True
    assert single_fill_gaps.buffering is False
    assert single_fill_gaps.fill_gaps is True


def test_get_rec_set_name(single_fill_gaps):
    assert ("/foo/bar", "") == single_fill_gaps.get_rec_set_name("/foo/bar/")
    assert ("/foo", "bar") == single_fill_gaps.get_rec_set_name("/foo/bar")
    assert ("/foo", "eye0_timestamp") == single_fill_gaps.get_rec_set_name(
        "/foo/eye0_timestamp.npy"
    )
