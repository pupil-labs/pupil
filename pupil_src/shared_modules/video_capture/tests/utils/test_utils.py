import os
import pytest
from video_capture.utils import RenameSet


@pytest.fixture(scope="function")
def tmp_dir(tmpdir_factory):
    TEMP_FILES = [
        # Audio files
        "audio_00010000.mp4", "audio_00010000.time",
        # ID0 files
        "Pupil Cam1 ID0.mp4", "Pupil Cam1 ID0.time",
        "Pupil Cam1 ID0_001.mp4", "Pupil Cam1 ID0_001.time",
        # ID2 files
        "Pupil Cam1 ID2.mp4", "Pupil Cam1 ID2.time",
        "Pupil Cam1 ID2_001.mp4", "Pupil Cam1 ID2_001.time",
        # Ohter files
        "key_00040000.key", "key_00040000.time",
        "imu_00020000.imu", "imu_00020000.time",
        ]
    tmp_dir = tmpdir_factory.mktemp("rename_data")
    for fn in TEMP_FILES:
        f = tmp_dir.join(fn)
        f.write("foobar"*2)
    return tmp_dir


def test_rename_file(tmp_dir):
    match_pattern = "*.time"
    rename_set = RenameSet(tmp_dir, match_pattern)
    # Rename to eye0
    rename_set.rename("Pupil Cam(0|1) ID0", "eye0")
    assert set(["eye0.mp4", "eye0_001.mp4"]).issubset(set(os.listdir(tmp_dir)))
    assert not any(
        elem in os.listdir(tmp_dir) for elem in
        ["Pupil Cam1 ID0.mp4", "Pupil Cam1 ID0_001.mp4"])
    # Rename to world
    rename_set.rename("Pupil Cam(0|1) ID2", "world")
    assert set(["world.mp4", "world_001.mp4"]).issubset(set(os.listdir(tmp_dir)))
    # Rename audio
    rename_set.rename("audio_[0-9]+", "audio")
    assert "audio.mp4" in os.listdir(tmp_dir)


def test_rewrite_timestamps(tmp_dir):
    match_pattern = "*.time"
    # Rewrite file end with time to npy timestamp file
    rewrite_time = RenameSet(tmp_dir, match_pattern, ['time'])
    rewrite_time.rewrite_time("_timestamps.npy")
    # Since we didn't rename the file, so the file name will stay the same
    assert all(elem in os.listdir(tmp_dir) for elem in [
            "audio_00010000_timestamps.npy", "Pupil Cam1 ID0_timestamps.npy",
            "Pupil Cam1 ID2_timestamps.npy"])


def test_load_intrinsics(tmp_dir):
    match_pattern = "*.time"
    rename_set = RenameSet(tmp_dir, match_pattern)
    # load_intrinsics
    assert "world.intrinsics" not in os.listdir(tmp_dir)
    rename_set.load_intrinsics()
    assert "world.intrinsics" in os.listdir(tmp_dir)
