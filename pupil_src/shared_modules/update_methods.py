"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import glob
import logging
import os
from shutil import copy2

import av
import numpy as np
from scipy.interpolate import interp1d

import csv_utils
import file_methods as fm
import player_methods as pm
from version_utils import VersionFormat, read_rec_version
from video_capture.utils import RenameSet

logger = logging.getLogger(__name__)


def update_recording_to_recent(rec_dir):

    meta_info = pm.load_meta_info(rec_dir)
    update_meta_info(rec_dir, meta_info)

    if (
        meta_info.get("Capture Software", "Pupil Capture") == "Pupil Mobile"
        and "Data Format Version" not in meta_info
    ):
        convert_pupil_mobile_recording_to_v094(rec_dir)
        meta_info["Data Format Version"] = "v0.9.4"
        update_meta_info(rec_dir, meta_info)

    # Reference format: v0.7.4
    rec_version = read_rec_version(meta_info)

    # Convert python2 to python3
    if rec_version <= VersionFormat("0.8.7"):
        update_recording_bytes_to_unicode(rec_dir)

    if rec_version >= VersionFormat("0.7.4"):
        pass
    elif rec_version >= VersionFormat("0.7.3"):
        update_recording_v073_to_v074(rec_dir)
    elif rec_version >= VersionFormat("0.5"):
        update_recording_v05_to_v074(rec_dir)
    elif rec_version >= VersionFormat("0.4"):
        update_recording_v04_to_v074(rec_dir)
    elif rec_version >= VersionFormat("0.3"):
        update_recording_v03_to_v074(rec_dir)
    else:
        logger.Error("This recording is too old. Sorry.")
        return

    # Incremental format updates
    if rec_version < VersionFormat("0.8.2"):
        update_recording_v074_to_v082(rec_dir)
    if rec_version < VersionFormat("0.8.3"):
        update_recording_v082_to_v083(rec_dir)
    if rec_version < VersionFormat("0.8.6"):
        update_recording_v083_to_v086(rec_dir)
    if rec_version < VersionFormat("0.8.7"):
        update_recording_v086_to_v087(rec_dir)
    if rec_version < VersionFormat("0.9.1"):
        update_recording_v087_to_v091(rec_dir)
    if rec_version < VersionFormat("0.9.3"):
        update_recording_v091_to_v093(rec_dir)
    if rec_version < VersionFormat("0.9.4"):
        update_recording_v093_to_v094(rec_dir)
    if rec_version < VersionFormat("0.9.13"):
        update_recording_v094_to_v0913(rec_dir)
    if rec_version < VersionFormat("1.3"):
        update_recording_v0913_to_v13(rec_dir)
    if rec_version < VersionFormat("1.4"):
        update_recording_v13_v14(rec_dir)

    # Do this independent of rec_version
    check_for_worldless_recording(rec_dir)

    if rec_version < VersionFormat("1.8"):
        update_recording_v14_v18(rec_dir)
    if rec_version < VersionFormat("1.9"):
        update_recording_v18_v19(rec_dir)

    # How to extend:
    # if rec_version < VersionFormat('FUTURE FORMAT'):
    #    update_recording_v081_to_FUTURE(rec_dir)


def update_meta_info(rec_dir, meta_info):
    logger.info("Updating meta info")
    meta_info_path = os.path.join(rec_dir, "info.csv")
    with open(meta_info_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_utils.write_key_value_file(csvfile, meta_info)


def _update_info_version_to(new_version, rec_dir):
    meta_info = pm.load_meta_info(rec_dir)
    meta_info["Data Format Version"] = new_version
    update_meta_info(rec_dir, meta_info)


def convert_pupil_mobile_recording_to_v094(rec_dir):
    logger.info("Converting Pupil Mobile recording to v0.9.4 format")
    # convert time files and rename corresponding videos
    match_pattern = "*.time"
    rename_set = RenameSet(rec_dir, match_pattern)
    rename_set.load_intrinsics()
    rename_set.rename("Pupil Cam([0-2]) ID0", "eye0")
    rename_set.rename("Pupil Cam([0-2]) ID1", "eye1")
    rename_set.rename("Pupil Cam([0-2]) ID2", "world")
    # Rewrite .time file to .npy file
    rewrite_time = RenameSet(rec_dir, match_pattern, ["time"])
    rewrite_time.rewrite_time("_timestamps.npy")
    pupil_data_loc = os.path.join(rec_dir, "pupil_data")
    if not os.path.exists(pupil_data_loc):
        logger.info('Creating "pupil_data"')
        fm.save_object(
            {"pupil_positions": [], "gaze_positions": [], "notifications": []},
            pupil_data_loc,
        )


def update_recording_v074_to_v082(rec_dir):
    _update_info_version_to("v0.8.2", rec_dir)


def update_recording_v082_to_v083(rec_dir):
    logger.info("Updating recording from v0.8.2 format to v0.8.3 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))

    for d in pupil_data["gaze_positions"]:
        if "base" in d:
            d["base_data"] = d.pop("base")

    fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))

    _update_info_version_to("v0.8.3", rec_dir)


def update_recording_v083_to_v086(rec_dir):
    logger.info("Updating recording from v0.8.3 format to v0.8.6 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))

    for topic in pupil_data.keys():
        for d in pupil_data[topic]:
            d["topic"] = topic

    fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))

    _update_info_version_to("v0.8.6", rec_dir)


def update_recording_v086_to_v087(rec_dir):
    logger.info("Updating recording from v0.8.6 format to v0.8.7 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))

    def _clamp_norm_point(pos):
        """realisitic numbers for norm pos should be in this range.
            Grossly bigger or smaller numbers are results bad exrapolation
            and can cause overflow erorr when denormalized and cast as int32.
        """
        return min(100.0, max(-100.0, pos[0])), min(100.0, max(-100.0, pos[1]))

    for g in pupil_data.get("gaze_positions", []):
        if "topic" not in g:
            # we missed this in one gaze mapper
            g["topic"] = "gaze"
        g["norm_pos"] = _clamp_norm_point(g["norm_pos"])

    fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))

    _update_info_version_to("v0.8.7", rec_dir)


def update_recording_v087_to_v091(rec_dir):
    logger.info("Updating recording from v0.8.7 format to v0.9.1 format")
    _update_info_version_to("v0.9.1", rec_dir)


def update_recording_v091_to_v093(rec_dir):
    logger.info("Updating recording from v0.9.1 format to v0.9.3 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))
    for g in pupil_data.get("gaze_positions", []):
        # fixing recordings made with bug https://github.com/pupil-labs/pupil/issues/598
        g["norm_pos"] = float(g["norm_pos"][0]), float(g["norm_pos"][1])

    fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))

    _update_info_version_to("v0.9.3", rec_dir)


def update_recording_v093_to_v094(rec_dir):
    logger.info("Updating recording from v0.9.3 to v0.9.4.")

    for file in os.listdir(rec_dir):
        if file.startswith(".") or os.path.splitext(file)[1] in (".mp4", ".avi"):
            continue
        rec_file = os.path.join(rec_dir, file)

        try:
            rec_object = fm.load_object(rec_file, allow_legacy=False)
            fm.save_object(rec_object, rec_file)
        except:
            try:
                rec_object = fm.load_object(rec_file, allow_legacy=True)
                fm.save_object(rec_object, rec_file)
                logger.info("Converted `{}` from pickle to msgpack".format(file))
            except:
                logger.warning("did not convert {}".format(rec_file))

    _update_info_version_to("v0.9.4", rec_dir)


def update_recording_v094_to_v0913(rec_dir, retry_on_averror=True):
    try:
        logger.info("Updating recording from v0.9.4 to v0.9.13")

        wav_file_loc = os.path.join(rec_dir, "audio.wav")
        aac_file_loc = os.path.join(rec_dir, "audio.mp4")
        audio_ts_loc = os.path.join(rec_dir, "audio_timestamps.npy")
        backup_ts_loc = os.path.join(rec_dir, "audio_timestamps_old.npy")
        if os.path.exists(wav_file_loc) and os.path.exists(audio_ts_loc):
            in_container = av.open(wav_file_loc)
            in_stream = in_container.streams.audio[0]
            in_frame_size = 0
            in_frame_num = 0

            out_container = av.open(aac_file_loc, "w")
            out_stream = out_container.add_stream("aac")

            for in_packet in in_container.demux():
                for audio_frame in in_packet.decode():
                    if not in_frame_size:
                        in_frame_size = audio_frame.samples
                    in_frame_num += 1
                    out_packet = out_stream.encode(audio_frame)
                    if out_packet is not None:
                        out_container.mux(out_packet)

            # flush encoder
            out_packet = out_stream.encode(None)
            while out_packet is not None:
                out_container.mux(out_packet)
                out_packet = out_stream.encode(None)

            out_frame_size = out_stream.frame_size
            out_frame_num = out_stream.frames
            out_frame_rate = out_stream.rate
            in_frame_rate = in_stream.rate

            out_container.close()

            old_ts = np.load(audio_ts_loc)
            np.save(backup_ts_loc, old_ts)

            if len(old_ts) != in_frame_num:
                in_frame_size /= len(old_ts) / in_frame_num
                logger.debug(
                    "Provided audio frame size is inconsistent with amount of timestamps. Correcting frame size to {}".format(
                        in_frame_size
                    )
                )

            old_ts_idx = (
                np.arange(0, len(old_ts) * in_frame_size, in_frame_size)
                * out_frame_rate
                / in_frame_rate
            )
            new_ts_idx = np.arange(0, out_frame_num * out_frame_size, out_frame_size)
            interpolate = interp1d(
                old_ts_idx, old_ts, bounds_error=False, fill_value="extrapolate"
            )
            new_ts = interpolate(new_ts_idx)

            # raise RuntimeError
            np.save(audio_ts_loc, new_ts)

        _update_info_version_to("v0.9.13", rec_dir)
    except av.AVError as averr:
        # Try to catch `libav.aac : Input contains (near) NaN/+-Inf` errors
        # Unfortunately, the above error is only logged not raised. Instead
        # `averr`, an `Invalid Argument` error with error number 22, is raised.
        if retry_on_averror and averr.errno == 22:
            # unfortunately
            logger.error("Encountered AVError. Retrying to update recording.")
            out_container.close()
            # Only retry once:
            update_recording_v094_to_v0913(rec_dir, retry_on_averror=False)
        else:
            raise  # re-raise exception


def update_recording_v0913_to_v13(rec_dir):
    logger.info("Updating recording from v0.9.13 to v1.3")

    # add notifications entry to pupil_data if missing
    pupil_data_loc = os.path.join(rec_dir, "pupil_data")
    pupil_data = fm.load_object(pupil_data_loc)
    if "notifications" not in pupil_data:
        pupil_data["notifications"] = []
        fm.save_object(pupil_data, pupil_data_loc)

    try:  # upgrade camera intrinsics
        old_calib_loc = os.path.join(rec_dir, "camera_calibration")
        old_calib = fm.load_object(old_calib_loc)
        res = tuple(old_calib["resolution"])
        del old_calib["resolution"]
        del old_calib["camera_name"]
        old_calib["cam_type"] = "radial"
        new_calib = {str(res): old_calib, "version": 1}
        fm.save_object(new_calib, os.path.join(rec_dir, "world.intrinsics"))
        logger.info("Replaced `camera_calibration` with `world.intrinsics`.")

        os.rename(old_calib_loc, old_calib_loc + ".deprecated")
    except IOError:
        pass

    _update_info_version_to("v1.3", rec_dir)


def update_recording_v13_v14(rec_dir):
    logger.info("Updating recording from v1.3 to v1.4")
    _update_info_version_to("v1.4", rec_dir)


def update_recording_v14_v18(rec_dir):
    logger.info("Updating recording from v1.4 to v1.8")
    legacy_topic_mapping = {
        "notifications": "notify",
        "gaze_positions": "gaze",
        "pupil_positions": "pupil",
    }

    with fm.Incremental_Legacy_Pupil_Data_Loader(rec_dir) as loader:
        for old_topic, values in loader.topic_values_pairs():
            new_topic = legacy_topic_mapping.get(old_topic, old_topic)
            with fm.PLData_Writer(rec_dir, new_topic) as writer:
                for datum in values:
                    if new_topic == "notify":
                        datum["topic"] = "notify." + datum["subject"]
                    elif new_topic == "pupil":
                        datum["topic"] += ".{}".format(datum["id"])
                    elif new_topic.startswith("surface"):
                        datum["topic"] = "surfaces." + datum["name"]
                    elif new_topic == "blinks" or new_topic == "fixations":
                        datum["topic"] += "s"

                    writer.append(datum)

    _update_info_version_to("v1.8", rec_dir)


def update_recording_v18_v19(rec_dir):
    logger.info("Updating recording from v1.8 to v1.9")

    def copy_cached_annotations():
        cache_dir = os.path.join(rec_dir, "offline_data")
        cache_file = os.path.join(cache_dir, "annotations.pldata")
        cache_ts_file = os.path.join(cache_dir, "annotations_timestamps.npy")
        annotation_file = os.path.join(rec_dir, "annotation.pldata")
        annotation_ts_file = os.path.join(rec_dir, "annotation_timestamps.npy")
        if os.path.exists(cache_file):
            logger.info("Version update: Copy annotations edited in Player.")
            copy2(cache_file, annotation_file)
            copy2(cache_ts_file, annotation_ts_file)

    def copy_recorded_annotations():
        logger.info("Version update: Copy recorded annotations.")
        notifications = fm.load_pldata_file(rec_dir, "notify")
        with fm.PLData_Writer(rec_dir, "annotation") as writer:
            for idx, topic in enumerate(notifications.topics):
                if topic == "notify.annotation":
                    annotation = notifications.data[idx]
                    ts = notifications.timestamps[idx]
                    writer.append_serialized(ts, "annotation", annotation.serialized)

    copy_cached_annotations()
    copy_recorded_annotations()

    _update_info_version_to("v1.9", rec_dir)


def check_for_worldless_recording(rec_dir):
    logger.info("Checking for world-less recording")
    valid_ext = (".mp4", ".mkv", ".avi", ".h264", ".mjpeg")

    world_video_exists = any(
        (
            os.path.splitext(f)[1] in valid_ext
            for f in glob.glob(os.path.join(rec_dir, "world.*"))
        )
    )

    if not world_video_exists:
        fake_world_version = 0
        fake_world_path = os.path.join(rec_dir, "world.fake")
        if os.path.exists(fake_world_path):
            fake_world = fm.load_object(fake_world_path)
            if fake_world["version"] == fake_world_version:
                return

        min_ts = np.inf
        max_ts = -np.inf
        for f in glob.glob(os.path.join(rec_dir, "eye*_timestamps.npy")):
            try:
                eye_ts = np.load(f)
                assert len(eye_ts.shape) == 1
                assert eye_ts.shape[0] > 1
                min_ts = min(min_ts, eye_ts[0])
                max_ts = max(max_ts, eye_ts[-1])
            except (FileNotFoundError, AssertionError):
                pass

        error_msg = "Could not generate world timestamps from eye timestamps. This is an invalid recording."
        assert -np.inf < min_ts < max_ts < np.inf, error_msg

        logger.warning("No world video found. Constructing an artificial replacement.")

        frame_rate = 30
        timestamps = np.arange(min_ts, max_ts, 1 / frame_rate)
        np.save(os.path.join(rec_dir, "world_timestamps.npy"), timestamps)
        fm.save_object(
            {
                "frame_rate": frame_rate,
                "frame_size": (1280, 720),
                "version": fake_world_version,
            },
            os.path.join(rec_dir, "world.fake"),
        )


def update_recording_bytes_to_unicode(rec_dir):
    logger.info("Updating recording from bytes to unicode.")

    def convert(data):
        if isinstance(data, bytes):
            return data.decode()
        elif isinstance(data, str) or isinstance(data, np.ndarray):
            return data
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    for file in os.listdir(rec_dir):
        if file.startswith(".") or os.path.splitext(file)[1] in (".mp4", ".avi"):
            continue
        rec_file = os.path.join(rec_dir, file)
        try:
            rec_object = fm.load_object(rec_file)
            converted_object = convert(rec_object)
            if converted_object != rec_object:
                logger.info("Converted `{}` from bytes to unicode".format(file))
                fm.save_object(converted_object, rec_file)
        except (fm.UnpicklingError, IsADirectoryError):
            continue

    # manually convert k v dicts.
    meta_info_path = os.path.join(rec_dir, "info.csv")
    with open(meta_info_path, "r", encoding="utf-8") as csvfile:
        meta_info = csv_utils.read_key_value_file(csvfile)
    with open(meta_info_path, "w", newline="") as csvfile:
        csv_utils.write_key_value_file(csvfile, meta_info)


def update_recording_v073_to_v074(rec_dir):
    logger.info("Updating recording from v0.7x format to v0.7.4 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))
    modified = False
    for p in pupil_data["pupil"]:
        if p["method"] == "3D c++":
            p["method"] = "3d c++"
            try:
                p["projected_sphere"] = p.pop("projectedSphere")
            except:
                p["projected_sphere"] = {"center": (0, 0), "angle": 0, "axes": (0, 0)}
            p["model_confidence"] = p.pop("modelConfidence")
            p["model_id"] = p.pop("modelID")
            p["circle_3d"] = p.pop("circle3D")
            p["diameter_3d"] = p.pop("diameter_3D")
            modified = True
    if modified:
        fm.save_object(
            fm.load_object(os.path.join(rec_dir, "pupil_data")),
            os.path.join(rec_dir, "pupil_data_old"),
        )
    try:
        fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass


def update_recording_v05_to_v074(rec_dir):
    logger.info("Updating recording from v0.5x/v0.6x/v0.7x format to v0.7.4 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))
    fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data_old"))
    for p in pupil_data["pupil"]:
        p["method"] = "2d python"
    try:
        fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass


def update_recording_v04_to_v074(rec_dir):
    logger.info("Updating recording from v0.4x format to v0.7.4 format")
    gaze_array = np.load(os.path.join(rec_dir, "gaze_positions.npy"))
    pupil_array = np.load(os.path.join(rec_dir, "pupil_positions.npy"))
    gaze_list = []
    pupil_list = []

    for datum in pupil_array:
        ts, confidence, id, x, y, diameter = datum[:6]
        pupil_list.append(
            {
                "timestamp": ts,
                "confidence": confidence,
                "id": id,
                "norm_pos": [x, y],
                "diameter": diameter,
                "method": "2d python",
                "ellipse": {"angle": 0.0, "center": [0.0, 0.0], "axes": [0.0, 0.0]},
            }
        )

    pupil_by_ts = dict([(p["timestamp"], p) for p in pupil_list])

    for datum in gaze_array:
        ts, confidence, x, y, = datum
        gaze_list.append(
            {
                "timestamp": ts,
                "confidence": confidence,
                "norm_pos": [x, y],
                "base": [pupil_by_ts.get(ts, None)],
            }
        )

    pupil_data = {"pupil_positions": pupil_list, "gaze_positions": gaze_list}
    try:
        fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass


def update_recording_v03_to_v074(rec_dir):
    logger.info("Updating recording from v0.3x format to v0.7.4 format")
    pupilgaze_array = np.load(os.path.join(rec_dir, "gaze_positions.npy"))
    gaze_list = []
    pupil_list = []

    for datum in pupilgaze_array:
        gaze_x, gaze_y, pupil_x, pupil_y, ts, confidence = datum
        # some bogus size and confidence as we did not save it back then
        pupil_list.append(
            {
                "timestamp": ts,
                "confidence": confidence,
                "id": 0,
                "norm_pos": [pupil_x, pupil_y],
                "diameter": 50,
                "method": "2d python",
            }
        )
        gaze_list.append(
            {
                "timestamp": ts,
                "confidence": confidence,
                "norm_pos": [gaze_x, gaze_y],
                "base": [pupil_list[-1]],
            }
        )

    pupil_data = {"pupil_positions": pupil_list, "gaze_positions": gaze_list}
    try:
        fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))
    except IOError:
        pass

    ts_path = os.path.join(rec_dir, "world_timestamps.npy")
    ts_path_old = os.path.join(rec_dir, "timestamps.npy")
    if not os.path.isfile(ts_path) and os.path.isfile(ts_path_old):
        os.rename(ts_path_old, ts_path)
