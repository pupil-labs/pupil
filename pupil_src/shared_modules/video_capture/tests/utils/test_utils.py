import os
from video_capture.utils import RenameSet


def test_rename_file(tmpdir):
    tmp_dir = tmpdir.mkdir("rename_data")
    file_lst = [
        # Audio file
        "audio_00010000.mp4", "audio_00010000.time",
        # ID0 file
        "Pupil Cam1 ID0.mp4", "Pupil Cam1 ID0.time",
        "Pupil Cam1 ID0_001.mp4", "Pupil Cam1 ID0_001.time",
        # ID2 file
        "Pupil Cam1 ID2.mp4", "Pupil Cam1 ID2.time",
        "Pupil Cam1 ID2_001.mp4", "Pupil Cam1 ID2_001.time",
        ]
    for fn in file_lst:
        f = tmp_dir.join(fn)
        f.write("foobar"*2)
    match_pattern = "*.time"
    rename_set = RenameSet(tmp_dir, match_pattern)
    # Rename to eye0
    rename_set.rename("Pupil Cam(0|1) ID0", "eye0")
    assert all(
        elem in os.listdir(tmp_dir) for elem in
        ["eye0.mp4", "eye0_001.mp4"])
    assert not any(
        elem in os.listdir(tmp_dir) for elem in
        ["Pupil Cam1 ID0.mp4", "Pupil Cam1 ID0_001.mp4"])
    # Rename to world
    rename_set.rename("Pupil Cam(0|1) ID2", "world")
    assert all(
        elem in os.listdir(tmp_dir) for elem in
        ["world.mp4", "world_001.mp4"])
    # Rename audio
    rename_set.rename("audio_[0-9]+", "audio")
    assert all(
        elem in os.listdir(tmp_dir) for elem in
        ["audio.mp4"])
    # Rewrite timestamp file
    rewrite_time = RenameSet(tmp_dir, match_pattern, ['time'])
    rewrite_time.rewrite_time("_timestamps.npy")
    assert all(
        elem in os.listdir(tmp_dir) for elem in [
            "eye0_timestamps.npy", "eye0_001_timestamps.npy",
            "world_timestamps.npy", "world_001_timestamps.npy",
            "audio_timestamps.npy"])
