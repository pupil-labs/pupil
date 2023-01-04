"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import datetime
import glob
import logging
import os
import uuid
from pathlib import Path
from shutil import copy2

import av
import csv_utils
import file_methods as fm
import numpy as np
from camera_models import Camera_Model
from scipy.interpolate import interp1d
from version_utils import parse_version

from .. import PupilRecording
from ..info import RecordingInfoFile
from ..info import recording_info_utils as rec_info_utils
from ..recording_utils import InvalidRecordingException

logger = logging.getLogger(__name__)


__all__ = ["transform_old_style_to_pprf_2_0"]


def transform_old_style_to_pprf_2_0(rec_dir: str) -> RecordingInfoFile:
    logger.info("Transform old style to new style recording...")
    _update_recording_to_old_style_v1_16(rec_dir)
    _generate_pprf_2_0_info_file(rec_dir)

    # rename info.csv file to info.old_style.csv
    logger.debug("Rename `info.csv` file to `info.old_style.csv`")
    info_csv = Path(rec_dir) / "info.csv"
    new_path = info_csv.with_name("info.old_style.csv")
    info_csv.replace(new_path)


def _generate_pprf_2_0_info_file(rec_dir):
    logger.debug("Generate PPRF 2.0 info file...")
    info_csv = rec_info_utils.read_info_csv_file(rec_dir)

    # Get information about recording from info.csv
    try:
        recording_uuid = info_csv.get("Recording UUID", uuid.uuid4())
        recording_software_name = info_csv.get(
            "Capture Software", RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_CAPTURE
        )
        start_time_system_s = float(
            info_csv.get(
                "Start Time (System)",
                _infer_start_time_system_from_legacy(info_csv, recording_software_name),
            )
        )
        start_time_synced_s = float(
            info_csv.get(
                "Start Time (Synced)", _infer_start_time_synced_from_legacy(rec_dir)
            )
        )
        duration_s = rec_info_utils.parse_duration_string(info_csv["Duration Time"])
        recording_software_version = info_csv["Capture Software Version"]
        recording_name = info_csv.get(
            "Recording Name", rec_info_utils.default_recording_name(rec_dir)
        )
        system_info = info_csv.get(
            "System Info", rec_info_utils.default_system_info(rec_dir)
        )
    except KeyError as e:
        logger.debug(f"KeyError while parsing old-style info.csv: {str(e)}")
        raise InvalidRecordingException(
            "This recording is too old to be opened with this version of Player!"
        )

    # Create a recording info file with the new format,
    # fill out the information, validate, and return.
    new_info_file = RecordingInfoFile.create_empty_file(
        rec_dir, fixed_version=parse_version("2.0")
    )
    new_info_file.recording_uuid = recording_uuid
    new_info_file.start_time_system_s = start_time_system_s
    new_info_file.start_time_synced_s = start_time_synced_s
    new_info_file.duration_s = duration_s
    new_info_file.recording_software_name = recording_software_name
    new_info_file.recording_software_version = recording_software_version
    new_info_file.recording_name = recording_name
    new_info_file.system_info = system_info
    new_info_file.validate()
    new_info_file.save_file()


########## PRIVATE ##########


def _update_recording_to_old_style_v1_16(rec_dir):
    logger.debug("Update old style recording...")

    meta_info = rec_info_utils.read_info_csv_file(rec_dir)
    update_meta_info(rec_dir, meta_info)

    # Reference format: v0.7.4
    rec_version = _read_rec_version_legacy(meta_info)

    # Convert python2 to python3
    if rec_version <= parse_version("0.8.7"):
        update_recording_bytes_to_unicode(rec_dir)

    if rec_version >= parse_version("0.7.4"):
        pass
    elif rec_version >= parse_version("0.7.3"):
        update_recording_v073_to_v074(rec_dir)
    elif rec_version >= parse_version("0.5"):
        update_recording_v05_to_v074(rec_dir)
    elif rec_version >= parse_version("0.4"):
        update_recording_v04_to_v074(rec_dir)
    elif rec_version >= parse_version("0.3"):
        update_recording_v03_to_v074(rec_dir)
    else:
        logger.error("This recording is too old. Sorry.")
        return

    # Incremental format updates
    if rec_version < parse_version("0.8.2"):
        update_recording_v074_to_v082(rec_dir)
    if rec_version < parse_version("0.8.3"):
        update_recording_v082_to_v083(rec_dir)
    if rec_version < parse_version("0.8.6"):
        update_recording_v083_to_v086(rec_dir)
    if rec_version < parse_version("0.8.7"):
        update_recording_v086_to_v087(rec_dir)
    if rec_version < parse_version("0.9.1"):
        update_recording_v087_to_v091(rec_dir)
    if rec_version < parse_version("0.9.3"):
        update_recording_v091_to_v093(rec_dir)
    if rec_version < parse_version("0.9.4"):
        update_recording_v093_to_v094(rec_dir)
    if rec_version < parse_version("0.9.13"):
        update_recording_v094_to_v0913(rec_dir)
    if rec_version < parse_version("1.3"):
        update_recording_v0913_to_v13(rec_dir)
    if rec_version < parse_version("1.4"):
        update_recording_v13_v14(rec_dir)

    if rec_version < parse_version("1.8"):
        update_recording_v14_v18(rec_dir)
    if rec_version < parse_version("1.9"):
        update_recording_v18_v19(rec_dir)
    if rec_version < parse_version("1.11"):
        update_recording_v19_v111(rec_dir)
    if rec_version < parse_version("1.13"):
        update_recording_v111_v113(rec_dir)
    if rec_version < parse_version("1.14"):
        update_recording_v113_v114(rec_dir)
    if rec_version < parse_version("1.16"):
        update_recording_v114_v116(rec_dir)

    # How to extend:
    # if rec_version < parse_version('FUTURE FORMAT'):
    #    update_recording_v081_to_FUTURE(rec_dir)


def update_meta_info(rec_dir, meta_info):
    # TODO: We read from PupilRecording here and save manually! I think this needs to be
    # adapted!
    logger.info("Updating meta info")
    meta_info_path = os.path.join(rec_dir, "info.csv")
    with open(meta_info_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_utils.write_key_value_file(csvfile, meta_info)


def _update_info_version_to(new_version, rec_dir):
    meta_info = rec_info_utils.read_info_csv_file(rec_dir)
    meta_info["Data Format Version"] = new_version
    update_meta_info(rec_dir, meta_info)


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
        except Exception:
            try:
                rec_object = fm.load_object(rec_file, allow_legacy=True)
                fm.save_object(rec_object, rec_file)
                logger.info(f"Converted `{file}` from pickle to msgpack")
            except Exception:
                logger.warning(f"did not convert {rec_file}")

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
                    "Provided audio frame size is inconsistent with amount of "
                    f"timestamps. Correcting frame size to {in_frame_size}"
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
    except OSError:
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


def update_recording_v19_v111(rec_dir):
    logger.info("Updating recording from v1.9 to v1.11")

    meta_info = rec_info_utils.read_info_csv_file(rec_dir)
    meta_info["Data Format Version"] = "v1.11"
    meta_info["Recording UUID"] = meta_info.get("Recording UUID", uuid.uuid4())
    update_meta_info(rec_dir, meta_info)


def update_recording_v111_v113(rec_dir):
    def undistort_vertices(verts, intrinsics):
        verts = np.asarray(verts)
        verts.shape = (4, 2)

        img_width, img_height = intrinsics.resolution
        target_width = int(0.8 * img_width)
        target_height = int(0.8 * img_height)

        verts_max = np.max(verts, axis=0)
        verts /= verts_max
        verts *= (target_width, target_height)

        verts += 100

        verts = intrinsics.undistort_points_on_image_plane(verts)

        verts -= 100

        verts /= (target_width, target_height)
        verts *= verts_max

        verts.shape = (4, 1, 2)
        verts = verts.tolist()

        return verts

    def make_update():
        surface_definitions_path = os.path.join(rec_dir, "surface_definitions")
        if not os.path.exists(surface_definitions_path):
            return

        surface_definitions_dict = fm.Persistent_Dict(surface_definitions_path)
        surface_definitions_backup_path = os.path.join(
            rec_dir, "surface_definitions_deprecated"
        )
        os.rename(surface_definitions_path, surface_definitions_backup_path)

        intrinsics_path = os.path.join(rec_dir, "world.intrinsics")
        if not os.path.exists(intrinsics_path):
            logger.warning(
                "Loading surface definitions failed: The data format of the "
                "surface definitions in this recording "
                "is too old and is no longer supported!"
            )
            return

        valid_ext = (".mp4", ".mkv", ".avi", ".h264", ".mjpeg")
        existing_videos = [
            f
            for f in glob.glob(os.path.join(rec_dir, "world.*"))
            if os.path.splitext(f)[1] in valid_ext
        ]
        if not existing_videos:
            return

        world_video_path = existing_videos[0]
        world_video = av.open(world_video_path, format=os.path.splitext()[-1][1:])
        f = world_video.streams.video[0].format
        resolution = f.width, f.height

        intrinsics = Camera_Model.from_file(rec_dir, "world", resolution)

        DEPRECATED_SQUARE_MARKER_KEY = "realtime_square_marker_surfaces"
        if DEPRECATED_SQUARE_MARKER_KEY not in surface_definitions_dict:
            return
        surfaces_definitions_old = surface_definitions_dict[
            DEPRECATED_SQUARE_MARKER_KEY
        ]

        surfaces_definitions_new = []
        for surface_def_old in surfaces_definitions_old:
            surface_def_new = {}
            surface_def_new["deprecated"] = True
            surface_def_new["name"] = surface_def_old["name"]
            surface_def_new["real_world_size"] = surface_def_old["real_world_size"]
            surface_def_new["build_up_status"] = 1.0

            reg_markers = []
            registered_markers_dist = []
            for id, verts in surface_def_old["markers"].items():
                reg_marker_dist = {"id": id, "verts_uv": verts}
                registered_markers_dist.append(reg_marker_dist)

                verts_undist = undistort_vertices(verts, intrinsics)
                reg_marker = {"id": id, "verts_uv": verts_undist}
                reg_markers.append(reg_marker)

            surface_def_new["registered_markers_dist"] = registered_markers_dist
            surface_def_new["reg_markers"] = reg_markers

            surfaces_definitions_new.append(surface_def_new)

        surface_definitions_dict_new = fm.Persistent_Dict(surface_definitions_path)
        surface_definitions_dict_new["surfaces"] = surfaces_definitions_new
        surface_definitions_dict_new.save()

    make_update()
    _update_info_version_to("v1.13", rec_dir)


def update_recording_v113_v114(rec_dir):
    _delete_all_lookup_files(rec_dir)
    _update_info_version_to("v1.14", rec_dir)


def update_recording_v114_v116(rec_dir):
    _delete_all_lookup_files(rec_dir)
    _update_info_version_to("v1.16", rec_dir)


def _delete_all_lookup_files(rec_dir):
    # Force re-build of video lookup tables
    names = ("world", "eye0", "eye1")
    rec_dir = Path(rec_dir)
    for name in names:
        try:
            (rec_dir / f"{name}_lookup.npy").unlink()
        except FileNotFoundError:
            pass


def update_recording_bytes_to_unicode(rec_dir):
    logger.info("Updating recording from bytes to unicode.")

    def convert(data):
        if isinstance(data, bytes):
            return data.decode()
        elif isinstance(data, str) or isinstance(data, np.ndarray):
            return data
        elif isinstance(data, collections.abc.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.abc.Iterable):
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
                logger.info(f"Converted `{file}` from bytes to unicode")
                fm.save_object(converted_object, rec_file)
        except (fm.UnpicklingError, IsADirectoryError):
            continue

    # manually convert k v dicts.
    meta_info_path = os.path.join(rec_dir, "info.csv")
    with open(meta_info_path, encoding="utf-8") as csvfile:
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
            except Exception:
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
    except OSError:
        pass


def update_recording_v05_to_v074(rec_dir):
    logger.info("Updating recording from v0.5x/v0.6x/v0.7x format to v0.7.4 format")
    pupil_data = fm.load_object(os.path.join(rec_dir, "pupil_data"))
    fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data_old"))
    for p in pupil_data["pupil"]:
        p["method"] = "2d python"
    try:
        fm.save_object(pupil_data, os.path.join(rec_dir, "pupil_data"))
    except OSError:
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

    pupil_by_ts = {p["timestamp"]: p for p in pupil_list}

    for datum in gaze_array:
        (
            ts,
            confidence,
            x,
            y,
        ) = datum
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
    except OSError:
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
    except OSError:
        pass

    ts_path = os.path.join(rec_dir, "world_timestamps.npy")
    ts_path_old = os.path.join(rec_dir, "timestamps.npy")
    if not os.path.isfile(ts_path) and os.path.isfile(ts_path_old):
        os.rename(ts_path_old, ts_path)


def _read_rec_version_legacy(meta_info):
    version_string = meta_info.get(
        "Data Format Version", meta_info["Capture Software Version"]
    )
    version_string = "".join(
        [c for c in version_string if c in "1234567890.-"]
    )  # strip letters in case of legacy version format
    logger.debug(f"Recording version: {version_string}")
    return parse_version(version_string)


def _infer_start_time_system_from_legacy(info_csv, recording_software_name):
    _warn_imprecise_value_inference()
    logger.warning(f"Missing meta info key: `Start Time (System)`.")

    # Read date and time from info_csv
    string_start_date = info_csv["Start Date"]
    string_start_time = info_csv["Start Time"]

    # Combine and parse to datetime.datetime
    string_start_date_time = f"{string_start_date} {string_start_time}"
    if (
        recording_software_name
        == RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_MOBILE
    ):
        format_date_time = "%d:%m:%Y %H:%M:%S"
    elif (
        recording_software_name
        == RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_CAPTURE
    ):
        format_date_time = "%d.%m.%Y %H:%M:%S"
    else:
        raise InvalidRecordingException(
            "Could not infer missing `Start Time (System)` value.\nUnexpected recording"
            f" software name: {recording_software_name}"
        )
    try:
        date_time = datetime.datetime.strptime(string_start_date_time, format_date_time)
    except ValueError as valerr:
        raise InvalidRecordingException(
            "Could not infer missing `Start Time (System)` value.\nUnexpected date time"
            f" input format: {string_start_date_time}"
        ) from valerr
    # Convert to Unix timestamp
    ts_start_date_time = date_time.timestamp()

    logger.info(f"Using {date_time} as input for `Start Time (System)` inference.")
    logger.info(f"Inferred `Start Time (System)`: {ts_start_date_time}")

    return ts_start_date_time


def _infer_start_time_synced_from_legacy(rec_dir):
    _warn_imprecise_value_inference()
    logger.warning(f"Missing meta info key: `Start Time (Synced)`.")

    files = PupilRecording.FileFilter(rec_dir)
    timestamp_files = files.core().timestamps()
    first_ts_per_timestamp_file = []
    for timestamp_file in timestamp_files:
        timestamps = np.load(str(timestamp_file))
        if timestamps.size == 0:
            continue
        first_ts_per_timestamp_file.append(timestamps[0])
        logger.info(f"First timestamp in {timestamp_file.name}: {timestamps[0]}")
    if not first_ts_per_timestamp_file:
        raise InvalidRecordingException(
            "Could not infer missing `Start Time (Synced)` value. No timestamps found."
        )
    inferred_start_time_synced = min(first_ts_per_timestamp_file)
    logger.info(f"Inferred `Start Time (Synced)`: {inferred_start_time_synced}")
    return inferred_start_time_synced


# global variable to warn only once
_SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE = True


def _warn_imprecise_value_inference():
    global _SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE
    if not _SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE:
        return
    logger.warning(
        "\n\n!! Deprecation Warning !! Pupil Mobile recordings recorded with older"
        " versions than r0.21.0, or Pupil Capture recordings recorded with older"
        " versions than v1.3, are deprecated and will not be supported by future"
        " Pupil Player versions!\n"
    )
    logger.warning(
        "\n\n!! Imprecise Value Inference !! In order to upgrade a deprecated"
        " recording, Pupil Player needs to infer missing meta data from the existing"
        " recording. This inference is imprecise and might cause issues when converting"
        " recorded Pupil time to wall clock time.\n"
    )
    _SHOULD_WARN_IMPRECISE_VALUE_INFERRENCE = False
