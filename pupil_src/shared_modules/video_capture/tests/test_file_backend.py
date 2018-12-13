import os
import logging
import pytest
import av
from multiprocessing import cpu_count
from video_capture.file_backend import File_Source
from video_capture.base_backend import EndofVideoError
from common import broken_data, multiple_data, single_data


@pytest.fixture
def single():
    '''Returns single data'''
    return File_Source(Empty(), source_path=single_data)


@pytest.fixture
def multiple():
    '''Returns multiple data'''
    return File_Source(Empty(), source_path=multiple_data)


@pytest.fixture
def broken():
    '''Returns broken data'''
    return File_Source(Empty(), source_path=broken_data)


class Empty:
    pass


def path_helper(data, name):
    return os.path.join(
        os.path.dirname(os.path.abspath(data)), name)


def test_get_conatiners_path(single, multiple, broken):
    '''
    Test _get_containers_path() function in different data set
    '''
    assert single._get_conatiners_path(single.source_path) == [
            path_helper(single_data, 'eye0.mp4')]
    assert multiple._get_conatiners_path(multiple.source_path) == [
            path_helper(multiple_data, 'eye0.mp4'),
            path_helper(multiple_data, 'eye0_001.mp4'),
            ]
    assert broken._get_conatiners_path(broken.source_path) == [
            path_helper(broken_data, 'eye0.mp4'),
            path_helper(broken_data, 'eye0_001.mp4'),
            path_helper(broken_data, 'eye0_002.mp4'),
            ]


def test_get_containers(single, broken, caplog):
    caplog.set_level(logging.INFO)
    # Normal case
    assert isinstance(
        single._get_containers(0), av.container.input.InputContainer)
    # No more container found
    with pytest.raises(EndofVideoError) as excinfo:
        single._get_containers(1)
    assert "Reached end of video file" in str(excinfo.value)
    # Auto skip broken file
    assert isinstance(
        broken._get_containers(1), av.container.input.InputContainer)
    assert ('Video at ' + path_helper(broken_data, 'eye0_001.mp4')
            + ' is broken' in caplog.text)


def test_get_streams(single, broken, caplog):
    caplog.set_level(logging.INFO)
    # Normal case
    stream = single._get_streams(single.container)
    assert isinstance(
        stream[0], av.video.stream.VideoStream)
    assert stream[0].thread_count == cpu_count()
    assert stream[1] is None


def test_get_timestamps_lst(single, multiple):
    assert single._get_pattern_lst(
            single.source_path, timestamps=True) == [
        path_helper(single_data, 'eye0_timestamps.npy'),
        ]
    assert multiple._get_pattern_lst(
            multiple.source_path, timestamps=True) == [
        path_helper(multiple_data, 'eye0_timestamps.npy'),
        path_helper(multiple_data, 'eye0_001_timestamps.npy'),
        ]


def test_set_timestamps(single):
    # Normal case
    single._set_timestamps(single.source_path)
    assert single.timestamps[0] == 2149.791964
