from unittest.mock import patch
import pytest
from video_capture.file_backend import File_Source


class Empty:
    pass


@pytest.mark.parametrize("prefix, suffix", [
    ("eye0", ".mp4"),
    ("eye1", ".mp4"),
    ("world", ".mjpeg"),
    ("Pupil Cam1 ID1", ".mp4"),
    ("Pupil Cam1 ID2", ".time"),
    ("world", ".npy"),
])
def test_get_container_lst(testdir, prefix, suffix):
    dic = {}
    # Timestamp file pattern
    if suffix == ".npy":
        timestamps = True
        dic[prefix + "_timestamps"] = "anything"
        # Create 4 tem file follow the name pattern
        for i in range(4, 0, -1):
            dic[prefix + "_00" + str(i) + "_timestamps"] = "anything"
    # Container file pattern
    else:
        timestamps = False
        dic[prefix] = "anything"
        for i in range(4, 0, -1):
            dic[prefix + "_00" + str(i)] = "anything"
    testdir.makefile(suffix, **dic)
    # Create other file in the tmp dir for testing
    testdir.makefile(suffix, other_file="anything")
    base = str(testdir) + "/" + prefix + suffix
    f = File_Source(Empty(), source_path=str(testdir) + "/" + prefix + suffix)
    if timestamps:
        container_lst = f._get_timestamps_lst(base)
    else:
        container_lst = f._get_container_lst(base)
    dic_lst = sorted([d + suffix for d in dic.keys()])
    assert container_lst == dic_lst
