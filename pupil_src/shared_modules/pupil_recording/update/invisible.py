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
import re
import shutil
import tempfile
import typing as T
from pathlib import Path

import av
import file_methods as fm
import methods as m
import numpy as np
import player_methods as pm
from version_utils import parse_version

from ..info import RecordingInfoFile
from ..info import recording_info_utils as utils
from ..recording import PupilRecording
from ..recording_utils import VALID_VIDEO_EXTENSIONS, InvalidRecordingException
from . import update_utils

logger = logging.getLogger(__name__)

NEWEST_SUPPORTED_VERSION = parse_version("1.4")


def transform_invisible_to_corresponding_new_style(rec_dir: str):
    logger.info("Transform Pupil Invisible to new style recording...")
    info_json = utils.read_info_json_file(rec_dir)
    pi_version = parse_version(info_json["data_format_version"])

    if pi_version > NEWEST_SUPPORTED_VERSION:
        raise InvalidRecordingException(
            f"This version of player is too old! Please upgrade."
        )

    # elif pi_version > 3.0:
    #     ...
    # elif pi_version > 2.0:
    #     ...

    else:
        _transform_invisible_v1_0_to_pprf_2_1(rec_dir)


def _transform_invisible_v1_0_to_pprf_2_1(rec_dir: str):
    _generate_pprf_2_1_info_file(rec_dir)

    # rename info.json file to info.invisible.json
    info_json = Path(rec_dir) / "info.json"
    new_path = info_json.with_name("info.invisible.json")
    info_json.replace(new_path)

    recording = PupilRecording(rec_dir)

    # Fix broken first frame issue, if affected.
    # This needs to happend before anything else
    # to make sure the rest of the pipeline is processed correctly.
    BrokenFirstFrameRecordingIssue.patch_recording_if_affected(recording)

    # patch world.intrinsics
    # NOTE: could still be worldless at this point
    update_utils._try_patch_world_instrinsics_file(
        rec_dir, recording.files().pi().world().videos()
    )

    _rename_pi_files(recording)
    _rewrite_timestamps(recording)
    _convert_gaze(recording)


def _generate_pprf_2_1_info_file(rec_dir: str) -> RecordingInfoFile:
    info_json = utils.read_info_json_file(rec_dir)

    # Get information about recording from info.csv and info.json
    recording_uuid = info_json["recording_id"]
    start_time_system_ns = int(info_json["start_time"])
    start_time_synced_ns = int(info_json["start_time"])
    duration_ns = int(info_json["duration"])
    recording_software_name = RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
    recording_software_version = info_json["app_version"]
    recording_name = utils.default_recording_name(rec_dir)
    system_info = android_system_info(info_json)

    # Create a recording info file with the new format,
    # fill out the information, validate, and return.
    new_info_file = RecordingInfoFile.create_empty_file(rec_dir, parse_version("2.1"))
    new_info_file.recording_uuid = recording_uuid
    new_info_file.start_time_system_ns = start_time_system_ns
    new_info_file.start_time_synced_ns = start_time_synced_ns
    new_info_file.duration_ns = duration_ns
    new_info_file.recording_software_name = recording_software_name
    new_info_file.recording_software_version = recording_software_version
    new_info_file.recording_name = recording_name
    new_info_file.system_info = system_info
    new_info_file.validate()
    new_info_file.save_file()


def _rename_pi_files(recording: PupilRecording):
    for pi_path, core_path in _pi_path_core_path_pairs(recording):
        pi_path.replace(core_path)  # rename with overwrite


def _pi_path_core_path_pairs(recording: PupilRecording):
    for pi_path in recording.files():
        # replace prefix based on cam_type, need to reformat part number
        match = re.match(
            r"^(?P<prefix>PI (?P<cam_type>left|right|world) v\d+ ps(?P<part>\d+))",
            pi_path.name,
        )
        if match:
            replacement_for_cam_type = {
                "right": "eye0",
                "left": "eye1",
                "world": "world",
            }
            replacement = replacement_for_cam_type[match.group("cam_type")]
            part_number = int(match.group("part"))
            if part_number > 1:
                # add zero-filled part number - 1
                # NOTE: recordings for PI start at part 1, mobile start at part 0
                replacement += f"_{part_number - 1:03}"

            core_name = pi_path.name.replace(match.group("prefix"), replacement)
            core_path = pi_path.with_name(core_name)
            yield pi_path, core_path


def _rewrite_timestamps(recording: PupilRecording):

    # Use start time from info file (instead of recording.meta_info.start_time_synced_ns)
    # to have a more precise value and avoid having a negative first timestamp when rewriting
    info_json = utils.read_pupil_invisible_info_file(recording.rec_dir)
    start_time_synced_ns = int(info_json["start_time"])

    def conversion(timestamps: np.array):
        # Subtract start_time from all times in the recording, so timestamps
        # start at 0. This is to increase precision when converting
        # timestamps to float32, e.g. for OpenGL!
        SECONDS_PER_NANOSECOND = 1e-9
        return (timestamps - start_time_synced_ns) * SECONDS_PER_NANOSECOND

    update_utils._rewrite_times(recording, dtype="<u8", conversion=conversion)


def _convert_gaze(recording: PupilRecording):
    width, height = 1088, 1080

    logger.info("Converting gaze data...")
    template_datum = {
        "topic": "gaze.pi",
        "norm_pos": None,
        "timestamp": None,
        "confidence": None,
    }
    with fm.PLData_Writer(recording.rec_dir, "gaze") as writer:
        for ((x, y), ts, conf) in pi_gaze_items(root_dir=recording.rec_dir):
            template_datum["timestamp"] = ts
            template_datum["norm_pos"] = m.normalize(
                (x, y), size=(width, height), flip_y=True
            )
            template_datum["confidence"] = conf
            writer.append(template_datum)
        logger.info(f"Converted {len(writer.ts_queue)} gaze positions.")


def android_system_info(info_json: dict) -> str:
    android_device_id = info_json.get("android_device_id", "?")
    android_device_name = info_json.get("android_device_name", "?")
    android_device_model = info_json.get("android_device_model", "?")
    return (
        f"Android device ID: {android_device_id}; "
        f"Android device name: {android_device_name}; "
        f"Android device model: {android_device_model}"
    )


class BrokenFirstFrameRecordingIssue:
    @classmethod
    def is_recording_affected(cls, recording: PupilRecording) -> bool:
        # If there are any world video and timestamps pairs affected - return True
        # Otherwise - False
        for _ in cls._pi_world_video_and_raw_time_affected_paths(recording):
            return True
        return False

    @classmethod
    def patch_recording_if_affected(cls, recording: PupilRecording):
        if not cls.is_recording_affected(recording=recording):
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            for v_path, t_path in cls._pi_world_video_and_raw_time_affected_paths(
                recording
            ):
                temp_t_path = Path(temp_dir) / t_path.name
                temp_v_path = Path(temp_dir) / v_path.name

                # Save video, dropping first frame, to temp file
                video_format = v_path.suffix[1:]
                in_container = av.open(str(v_path), format=video_format)
                out_container = av.open(str(temp_v_path), "w", format=video_format)

                # input -> output stream mapping
                stream_mapping = {
                    in_stream: out_container.add_stream(template=in_stream)
                    for in_stream in in_container.streams
                }

                # Keep track of streams that should skip frame
                stream_should_skip_frame = {
                    in_stream: in_stream.codec_context.type == "video"
                    or in_stream.codec_context.type == "audio"
                    for in_stream in stream_mapping.keys()
                }

                for packet in in_container.demux():
                    if stream_should_skip_frame[packet.stream]:
                        # Once the stream skipped the first frame, don't skip anymore
                        stream_should_skip_frame[packet.stream] = False
                        continue
                    packet.stream = stream_mapping[packet.stream]
                    out_container.mux(packet)
                out_container.close()

                # Save raw time file, dropping first timestamp, to temp file
                ts = cls._pi_raw_time_load(t_path)
                cls._pi_raw_time_save(temp_t_path, ts[1:])

                # Overwrite old files with new ones
                v_path = v_path.with_name(v_path.stem).with_suffix(v_path.suffix)

                # pathlib.Path.replace raises an `OSError: [Errno 18] Cross-device link`
                # if the temp file is on a different device than the original. This
                # https://stackoverflow.com/a/43967659/5859392 recommends using
                # shutil.move instead (only supports pathlike in python>=3.9).
                shutil.move(str(temp_v_path), str(v_path))
                shutil.move(str(temp_t_path), str(t_path))

    @classmethod
    def _pi_world_video_and_raw_time_affected_paths(cls, recording: PupilRecording):
        # Check if the first timestamp is greater than the second timestamp from world timestamps;
        # this is a symptom of Pupil Invisible recording with broken first frame.
        # If the first timestamp is greater, remove it from the timestamps and overwrite the file.
        for v_path, ts_path in cls._pi_world_video_and_raw_time_paths(recording):

            try:
                with av.open(str(v_path), format=v_path.suffix[1:]) as in_container:
                    packets = in_container.demux(video=0)

                    # Try to demux the first frame.
                    # This is expected to raise an error.
                    # If no error is raised, ignore this video.
                    try:
                        _ = next(packets).decode()
                    except av.AVError:
                        pass  # Expected
                    except StopIteration:
                        continue  # Not expected
                    else:
                        continue  # Not expected

                    # Try to demux the second frame.
                    # This is not expected to raise an error.
                    # If an error is raised, ignore this video.
                    try:
                        _ = next(packets).decode()
                    except av.AVError:
                        continue  # Not expected
                    except StopIteration:
                        continue  # Not expected
                    else:
                        pass  # Expected

                    # Check there are 2 or more raw timestamps.
                    raw_time = cls._pi_raw_time_load(ts_path)
                    if len(raw_time) < 2:
                        continue

                    yield v_path, ts_path
            except av.AVError:
                logger.exception(
                    f"Encountered an issue while reading {v_path}. "
                    "Skipping file. Transformed recording might be incomplete."
                )

    @classmethod
    def _pi_world_video_and_raw_time_paths(cls, recording: PupilRecording):
        for pi_path, core_path in _pi_path_core_path_pairs(recording):
            if not cls._is_pi_world_video_path(pi_path):
                continue

            video_path = pi_path
            raw_time_path = video_path.with_suffix(".time")

            assert raw_time_path.is_file(), f"Expected file at path: {raw_time_path}"

            yield video_path, raw_time_path

    @staticmethod
    def _is_pi_world_video_path(path):
        def match_any(target, *patterns):
            return any([re.search(pattern, str(target)) for pattern in patterns])

        is_pi_world = match_any(path.name, r"^PI world v(\d+) ps(\d+)")

        is_video = match_any(
            path.name, *[rf"\.{ext}$" for ext in VALID_VIDEO_EXTENSIONS]
        )

        return is_pi_world and is_video

    @staticmethod
    def _pi_raw_time_load(path):
        return np.fromfile(str(path), dtype="<u8")

    @staticmethod
    def _pi_raw_time_save(path, arr):
        arr.tofile(str(path))


def pi_gaze_items(root_dir):
    """Yields one (location, timestamp, confidence) triplet for each gaze point

    Pupil Invisible Companion records this information into three different sets of
    files. Their names can be matched by the following regex patterns:
        - `^gaze ps[0-9]+.raw$`
        - `^gaze ps[0-9]+.time$`
        - `^worn ps[0-9]+.raw$`

    The worn data is a stream of values of either 0 or 255, indicating that the glasses
    were (not) worn. Pupil Player maps these to gaze confidence values of 0.0 and 1.0
    respectively.

    Since all `*.time` files are converted to Pupil Player before this function is being
    called, we match the `^gaze ps[0-9]+_timestamps.npy$` pattern on the recording files
    instead. When looking for the location and worn data, the function just replaces the
    necessary parts of the timestamp file names instead of performing separate regex
    matches.

    If the recording was successfully post-processed and downloaded from Pupil Cloud, it
    will contain 200Hz-densified gaze data. This data replaces the real-time recorded
    data by Pupil Invisible Companion and is stored in three files:
        - `gaze_200hz.raw`
        - `gaze_200hz.time` (or `gaze_200hz_timestamps.npy` if upgraded)
        - `worn_200hz.raw`

    The worn data is a special case as it was introduced at different points in time to
    Pupil Invisible Companion and Pupil Cloud. In other words, it is possible that there
    is no worn data, only real-time recorded worn data, or 200 Hz worn data. The latter
    is preferred. If 200 Hz gaze data is only available with real-time recorded worn
    data, the latter is interpolated to 200 Hz using a k-nearest-neighbour (k=1)
    approach. If no worn data is available, or the numbers of worn samples and gaze
    timestamps are not consistent, Pupil Player assumes a confidence value of 1.0 for
    every gaze point.
    """
    root_dir = Path(root_dir)
    # This pattern will match any filename that:
    # - starts with "gaze ps"
    # - is followed by one or more digits
    # - ends with "_timestamps.npy"
    timestamps_realtime_pattern = r"^gaze ps[0-9]+_timestamps.npy$"
    timestamps_realtime_paths = matched_files_by_name_pattern(
        root_dir, timestamps_realtime_pattern
    )
    # Use 200hz data only if both gaze data and timestamps are available at 200hz
    raw_200hz_path = _find_raw_200hz_path(root_dir)
    timestamps_200hz_path = _find_timestamps_200hz_path(root_dir)
    if raw_200hz_path and timestamps_200hz_path:
        worn_200hz_path = _find_worn_200hz_path(root_dir)
        yield from _pi_posthoc_200hz_gaze_items(
            raw_200hz_path,
            timestamps_200hz_path,
            worn_200hz_path,
            timestamps_realtime_paths,
        )
    else:
        yield from _pi_realtime_recorded_gaze_items(timestamps_realtime_paths)


def _pi_posthoc_200hz_gaze_items(
    raw_200hz_path, timestamps_200hz_path, worn_200hz_path, timestamps_realtime_paths
):
    raw_data = _load_raw_data(raw_200hz_path)
    timestamps = _load_timestamps_data(timestamps_200hz_path)

    if worn_200hz_path is not None:
        conf_data = _load_worn_data(worn_200hz_path)
    else:
        conf_data = _find_and_load_densified_worn_data(
            timestamps, timestamps_realtime_paths
        )

    raw_data, timestamps = _equalize_length_if_necessary(raw_data, timestamps)
    conf_data = _validated_conf_data(conf_data, timestamps)
    yield from zip(raw_data, timestamps, conf_data)


def _pi_realtime_recorded_gaze_items(timestamps_realtime_paths):
    for timestamps_path in timestamps_realtime_paths:
        raw_data = _load_raw_data(_find_raw_path(timestamps_path))
        timestamps = _load_timestamps_data(timestamps_path)
        conf_data = _load_worn_data(_find_worn_path(timestamps_path))

        raw_data, timestamps = _equalize_length_if_necessary(raw_data, timestamps)
        conf_data = _validated_conf_data(conf_data, timestamps)
        yield from zip(raw_data, timestamps, conf_data)


def _find_timestamps_200hz_path(root_dir: Path):
    path = root_dir / "gaze_200hz_timestamps.npy"
    if path.is_file():
        return path
    else:
        return None


def _find_raw_200hz_path(root_dir: Path):
    path = root_dir / "gaze_200hz.raw"
    if path.is_file():
        return path
    else:
        return None


def _find_worn_200hz_path(root_dir: Path):
    path = root_dir / "worn_200hz.raw"
    if path.is_file():
        return path
    else:
        return None


def _find_raw_path(timestamps_path: Path):
    name = timestamps_path.name.replace("_timestamps", "")
    path = timestamps_path.with_name(name).with_suffix(".raw")
    assert path.is_file(), f"The file does not exist at path: {path}"
    return path


def _find_worn_path(timestamps_path: Path):
    name = timestamps_path.name
    name = name.replace("gaze", "worn")
    name = name.replace("_timestamps", "")
    path = timestamps_path.with_name(name).with_suffix(".raw")
    if path.is_file():
        return path
    else:
        return None


def _load_timestamps_data(path):
    timestamps = np.load(str(path))
    return timestamps


def _load_raw_data(path):
    raw_data = np.fromfile(str(path), "<f4")
    raw_data_dtype = raw_data.dtype
    raw_data.shape = (-1, 2)
    return np.asarray(raw_data, dtype=raw_data_dtype)


def _load_worn_data(path: Path):
    if not (path and path.exists()):
        return None

    confidences = np.fromfile(str(path), "<u1") / 255.0
    return np.clip(confidences, 0.0, 1.0)


def _find_and_load_densified_worn_data(
    timestamps_200hz, timestamps_realtime_paths: T.List[Path]
):
    if not timestamps_realtime_paths:
        return None
    # Load and densify confidence data when 200hz gaze is available, but only
    # non-200hz confidence is available
    conf_data, timestamps_realtime = _find_and_load_realtime_recorded_worn_data(
        timestamps_realtime_paths
    )
    densification_idc = pm.find_closest(timestamps_realtime, timestamps_200hz)
    return conf_data[densification_idc]


def _find_and_load_realtime_recorded_worn_data(timestamps_realtime_paths: T.List[Path]):
    # assumes at least one path in `timestamps_realtime_paths`, otherwise np.concatenate
    # will raise ValueError: need at least one array to concatenate
    assert (
        len(timestamps_realtime_paths) > 0
    ), "Requires at least one real-time recorded gaze timestamp path"
    conf_all = []
    ts_all = []
    for timestamps_path in timestamps_realtime_paths:
        ts = _load_timestamps_data(timestamps_path)
        conf_data = _load_worn_data(_find_worn_path(timestamps_path))
        conf_data = _validated_conf_data(conf_data, ts)
        conf_all.append(conf_data)
        ts_all.append(ts)
    conf_all = np.concatenate(conf_all)
    ts_all = np.concatenate(ts_all)
    return conf_all, ts_all


def _equalize_length_if_necessary(raw_data, timestamps):
    if len(raw_data) != len(timestamps):
        logger.warning(
            f"There is a mismatch between the number of raw data ({len(raw_data)}) "
            f"and the number of timestamps ({len(timestamps)})!"
        )
        size = min(len(raw_data), len(timestamps))
        raw_data = raw_data[:size]
        timestamps = timestamps[:size]
    return raw_data, timestamps


def _validated_conf_data(conf_data, timestamps):
    if conf_data is not None and len(conf_data) != len(timestamps):
        logger.warning(
            "There is a mismatch between the number of confidence data "
            f"({len(conf_data)}) and the number of timestamps ({len(timestamps)})! "
            "Not using confidence data."
        )
        conf_data = None
    if conf_data is None:
        conf_data = np.ones(len(timestamps))
    return conf_data


def matched_files_by_name_pattern(parent_dir: Path, name_pattern: str) -> T.List[Path]:
    # Get all non-recursive directory contents
    contents = filter(Path.is_file, parent_dir.iterdir())
    # Filter content that matches the name by regex pattern
    return sorted(c for c in contents if re.match(name_pattern, c.name) is not None)
