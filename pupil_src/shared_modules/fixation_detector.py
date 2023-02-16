"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
"""
Fixations general knowledge from literature review
    + Goldberg et al. - fixations rarely < 100ms and range between 200ms and 400ms in
      duration (Irwin, 1992 - fixations dependent on task between 150ms - 600ms)
    + Very short fixations are considered not meaningful for studying behavior
        - eye+brain require time for info to be registered (see Munn et al. APGV, 2008)
    + Fixations are rarely longer than 800ms in duration
        + Smooth Pursuit is exception and different motif
        + If we do not set a maximum duration, we will also detect smooth pursuit (which
          is acceptable since we compensate for VOR)
Terms
    + dispersion (spatial) = how much spatial movement is allowed within one fixation
      (in visual angular degrees or pixels)
    + duration (temporal) = what is the minimum time required for gaze data to be within
      dispersion threshold?
"""

import csv
import enum
import logging
import os
import typing as T
from bisect import bisect_left, bisect_right
from collections import deque
from types import SimpleNamespace

import background_helper as bh
import cv2
import data_changed
import file_methods as fm
import msgpack
import numpy as np
import player_methods as pm
from hotkey import Hotkey
from methods import denormalize
from observable import Observable
from plugin import Plugin
from pupil_recording import PupilRecording, RecordingInfo
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle
from pyglui.pyfontstash import fontstash
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


class FixationDetectionMethod(enum.Enum):
    GAZE_2D = "2d gaze"
    GAZE_3D = "3d gaze"


class Fixation_Detector_Base(Plugin):
    icon_chr = chr(0xEC03)
    icon_font = "pupil_icons"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Fixation Detector"


def fixation_from_data(
    dispersion: float,
    method: FixationDetectionMethod,
    base_data: T.Iterable,
    timestamps=None,
):
    norm_pos = np.mean([gp["norm_pos"] for gp in base_data], axis=0).tolist()
    dispersion = np.rad2deg(dispersion)  # in degrees

    fix = {
        "topic": "fixations",
        "norm_pos": norm_pos,
        "dispersion": dispersion,
        "method": method.value,
        "base_data": list(base_data),
        "timestamp": base_data[0]["timestamp"],
        "duration": (base_data[-1]["timestamp"] - base_data[0]["timestamp"]) * 1000,
        "confidence": float(np.mean([gp["confidence"] for gp in base_data])),
    }
    if method == FixationDetectionMethod.GAZE_3D:
        fix["gaze_point_3d"] = np.mean(
            [gp["gaze_point_3d"] for gp in base_data if "gaze_point_3d" in gp], axis=0
        ).tolist()
    if timestamps is not None:
        start, end = base_data[0]["timestamp"], base_data[-1]["timestamp"]
        start, end = np.searchsorted(timestamps, [start, end])
        end = min(end, len(timestamps) - 1)  # fix `list index out of range` error
        fix["start_frame_index"] = int(start)
        fix["end_frame_index"] = int(end)
        fix["mid_frame_index"] = int((start + end) // 2)
    return fix


class Fixation_Result_Factory:
    __slots__ = ("_id_counter",)

    def __init__(self):
        self._id_counter = 0

    def from_data(self, *args, **kwargs):
        datum = fixation_from_data(*args, **kwargs)
        self._set_fixation_id(datum)
        fixation_start = datum["timestamp"]
        fixation_stop = fixation_start + (datum["duration"] / 1000)
        datum = self._serialize(datum)
        return (datum, fixation_start, fixation_stop)

    def _set_fixation_id(self, fixation):
        fixation["id"] = self._id_counter
        self._id_counter += 1

    def _serialize(self, fixation):
        serialization_hook = fm.Serialized_Dict.packing_hook
        fixation_serialized = msgpack.packb(
            fixation, use_bin_type=True, default=serialization_hook
        )
        return fixation_serialized


def vector_dispersion(vectors):
    distances = pdist(vectors, metric="cosine")
    dispersion = np.arccos(1.0 - distances.max())
    return dispersion


def gaze_dispersion(capture, gaze_subset, method: FixationDetectionMethod) -> float:
    if method is FixationDetectionMethod.GAZE_3D:
        vectors = np.array([gp["gaze_point_3d"] for gp in gaze_subset])
    elif method is FixationDetectionMethod.GAZE_2D:
        locations = np.array([gp["norm_pos"] for gp in gaze_subset])

        # denormalize
        width, height = capture.frame_size
        locations[:, 0] *= width
        locations[:, 1] = (1.0 - locations[:, 1]) * height

        # undistort onto 3d plane
        vectors = capture.intrinsics.unprojectPoints(locations)
    else:
        raise ValueError(f"Unknown method '{method}'")

    dist = vector_dispersion(vectors)
    return dist


def can_use_3d_gaze_mapping(gaze_data) -> bool:
    return all("gaze_point_3d" in gp for gp in gaze_data)


def detect_fixations(
    capture, gaze_data, max_dispersion, min_duration, max_duration, min_data_confidence
):
    yield "Detecting fixations...", ()
    gaze_data = (
        fm.Serialized_Dict(msgpack_bytes=serialized) for serialized in gaze_data
    )
    gaze_data = [
        datum for datum in gaze_data if datum["confidence"] > min_data_confidence
    ]
    if not gaze_data:
        logger.warning("No data available to find fixations")
        return "Fixation detection failed", ()

    method = (
        FixationDetectionMethod.GAZE_3D
        if can_use_3d_gaze_mapping(gaze_data)
        else FixationDetectionMethod.GAZE_2D
    )
    logger.info(f"Starting fixation detection using {method.value} data...")
    fixation_result = Fixation_Result_Factory()

    working_queue = deque()
    remaining_gaze = deque(gaze_data)

    while remaining_gaze:
        # check if working_queue contains enough data
        if (
            len(working_queue) < 2
            or (working_queue[-1]["timestamp"] - working_queue[0]["timestamp"])
            < min_duration
        ):
            datum = remaining_gaze.popleft()
            working_queue.append(datum)
            continue

        # min duration reached, check for fixation
        dispersion = gaze_dispersion(capture, working_queue, method)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            working_queue.popleft()
            continue

        left_idx = len(working_queue)

        # minimal fixation found. collect maximal data
        # to perform binary search for fixation end
        while remaining_gaze:
            datum = remaining_gaze[0]
            if datum["timestamp"] > working_queue[0]["timestamp"] + max_duration:
                break  # maximum data found
            working_queue.append(remaining_gaze.popleft())

        # check for fixation with maximum duration
        dispersion = gaze_dispersion(capture, working_queue, method)
        if dispersion <= max_dispersion:
            fixation = fixation_result.from_data(
                dispersion, method, working_queue, capture.timestamps
            )
            yield "Detecting fixations...", fixation
            working_queue.clear()  # discard old Q
            continue

        slicable = list(working_queue)  # deque does not support slicing
        right_idx = len(working_queue)

        # binary search
        while left_idx < right_idx - 1:
            middle_idx = (left_idx + right_idx) // 2
            dispersion = gaze_dispersion(
                capture,
                slicable[: middle_idx + 1],
                method,
            )
            if dispersion <= max_dispersion:
                left_idx = middle_idx
            else:
                right_idx = middle_idx

        # left_idx-1 is last valid base datum
        final_base_data = slicable[:left_idx]
        to_be_placed_back = slicable[left_idx:]
        dispersion_result = gaze_dispersion(capture, final_base_data, method)

        fixation = fixation_result.from_data(
            dispersion_result, method, final_base_data, capture.timestamps
        )
        yield "Detecting fixations...", fixation
        working_queue.clear()  # clear queue
        remaining_gaze.extendleft(reversed(to_be_placed_back))

    yield "Fixation detection complete", ()


class Offline_Fixation_Detector(Observable, Fixation_Detector_Base):
    """Dispersion-duration-based fixation detector.

    This plugin detects fixations based on a dispersion threshold in terms of
    degrees of visual angle within a given duration window. It tries to maximize
    the length of classified fixations within the duration window, e.g. instead
    of creating two consecutive fixations of length 300 ms it creates a single
    fixation with length 600 ms. Fixations do not overlap. Binary search is used
    to find the correct fixation length within the duration window.

    If 3d pupil data is available the fixation dispersion will be calculated
    based on the positional angle of the eye. These fixations have their method
    field set to "pupil". If no 3d pupil data is available the plugin will
    assume that the gaze data is calibrated and calculate the dispersion in
    visual angle within the coordinate system of the world camera. These
    fixations will have their method field set to "gaze".
    """

    CACHE_VERSION = 1

    class VersionMismatchError(ValueError):
        pass

    class ConfigMismatchError(ValueError):
        pass

    class DataMismatchError(ValueError):
        pass

    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # Disable fixation detector in Player if Pupil Invisible recording
                return False
        return super().is_available_within_context(g_pool)

    def __init__(
        self,
        g_pool,
        max_dispersion=1.50,
        min_duration=80,
        max_duration=220,
        show_fixations=True,
    ):
        super().__init__(g_pool)
        self.max_dispersion = max_dispersion
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.show_fixations = show_fixations
        self.current_fixation_details = None
        self.fixation_data = []
        self.prev_index = -1
        self.bg_task = None
        self.status = ""
        self.data_dir = os.path.join(g_pool.rec_dir, "offline_data")
        self._gaze_changed_listener = data_changed.Listener(
            "gaze_positions", g_pool.rec_dir, plugin=self
        )
        self._gaze_changed_listener.add_observer("on_data_changed", self._classify)
        self._fixations_changed_announcer = data_changed.Announcer(
            "fixations", g_pool.rec_dir, plugin=self
        )
        try:
            self.load_offline_data()
        except (
            FileNotFoundError,
            self.VersionMismatchError,
            self.ConfigMismatchError,
            self.DataMismatchError,
        ) as err:
            logger.debug(f"Offline data not loaded: {err} ({type(err).__name__})")
            self.notify_all(
                {"subject": "fixation_detector.should_recalculate", "delay": 0.5}
            )

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Fixation Detector"

        def set_max_dispersion(new_value):
            self.max_dispersion = new_value
            self.notify_all(
                {"subject": "fixation_detector.should_recalculate", "delay": 1.0}
            )

        def set_min_duration(new_value):
            self.min_duration = min(new_value, self.max_duration)
            self.notify_all(
                {"subject": "fixation_detector.should_recalculate", "delay": 1.0}
            )

        def set_max_duration(new_value):
            self.max_duration = max(new_value, self.min_duration)
            self.notify_all(
                {"subject": "fixation_detector.should_recalculate", "delay": 1.0}
            )

        def jump_next_fixation(_):
            cur_idx = self.last_frame_idx
            all_idc = [f["mid_frame_index"] for f in self.g_pool.fixations]
            if not all_idc:
                logger.warning("No fixations available")
                return
            # wrap-around index
            tar_fix = bisect_right(all_idc, cur_idx) % len(all_idc)
            self.notify_all(
                {
                    "subject": "seek_control.should_seek",
                    "index": int(self.g_pool.fixations[tar_fix]["mid_frame_index"]),
                }
            )

        def jump_prev_fixation(_):
            cur_idx = self.last_frame_idx
            all_idc = [f["mid_frame_index"] for f in self.g_pool.fixations]
            if not all_idc:
                logger.warning("No fixations available")
                return
            # wrap-around index
            tar_fix = (bisect_left(all_idc, cur_idx) - 1) % len(all_idc)
            self.notify_all(
                {
                    "subject": "seek_control.should_seek",
                    "index": int(self.g_pool.fixations[tar_fix]["mid_frame_index"]),
                }
            )

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(ui.Info_Text(help_str))
        self.menu.append(
            ui.Info_Text(
                "To start the export, wait until the detection has finished and press "
                "the export button or type 'e'."
            )
        )
        self.menu.append(
            ui.Info_Text(
                "Note: This plugin does not process fixations that have been calculated"
                " and recorded in real time."
            )
        )

        self.menu.append(
            ui.Slider(
                "max_dispersion",
                self,
                min=0.01,
                step=0.1,
                max=5.0,
                label="Maximum Dispersion [degrees]",
                setter=set_max_dispersion,
            )
        )
        self.menu.append(
            ui.Slider(
                "min_duration",
                self,
                min=10,
                step=10,
                max=4000,
                label="Minimum Duration [milliseconds]",
                setter=set_min_duration,
            )
        )
        self.menu.append(
            ui.Slider(
                "max_duration",
                self,
                min=10,
                step=10,
                max=4000,
                label="Maximum Duration [milliseconds]",
                setter=set_max_duration,
            )
        )
        self.menu.append(
            ui.Text_Input(
                "status", self, label="Detection progress:", setter=lambda x: None
            )
        )
        self.menu.append(ui.Switch("show_fixations", self, label="Show fixations"))
        self.current_fixation_details = ui.Info_Text("")
        self.menu.append(self.current_fixation_details)

        self.next_fix_button = ui.Thumb(
            "jump_next_fixation",
            setter=jump_next_fixation,
            getter=lambda: False,
            label=chr(0xE044),
            hotkey=Hotkey.FIXATION_NEXT_PLAYER_HOTKEY(),
            label_font="pupil_icons",
        )
        self.next_fix_button.status_text = "Next Fixation"
        self.g_pool.quickbar.append(self.next_fix_button)

        self.prev_fix_button = ui.Thumb(
            "jump_prev_fixation",
            setter=jump_prev_fixation,
            getter=lambda: False,
            label=chr(0xE045),
            hotkey=Hotkey.FIXATION_PREV_PLAYER_HOTKEY(),
            label_font="pupil_icons",
        )
        self.prev_fix_button.status_text = "Previous Fixation"
        self.g_pool.quickbar.append(self.prev_fix_button)

    def deinit_ui(self):
        self.remove_menu()
        self.current_fixation_details = None
        self.g_pool.quickbar.remove(self.next_fix_button)
        self.g_pool.quickbar.remove(self.prev_fix_button)
        self.next_fix_button = None
        self.prev_fix_button = None

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None

    def get_init_dict(self):
        return {
            "max_dispersion": self.max_dispersion,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "show_fixations": self.show_fixations,
        }

    def on_notify(self, notification):
        if notification["subject"] == "min_data_confidence_changed":
            logger.info("Minimal data confidence changed. Recalculating.")
            self._classify()
        elif notification["subject"] == "fixation_detector.should_recalculate":
            self._classify()
        elif notification["subject"] == "should_export":
            self.export_fixations(notification["ts_window"], notification["export_dir"])

    def _classify(self):
        """
        classify fixations
        """
        if self.g_pool.app == "exporter":
            return

        if self.bg_task:
            self.bg_task.cancel()

        gaze_data = [gp.serialized for gp in self.g_pool.gaze_positions]

        cap = SimpleNamespace()
        cap.frame_size = self.g_pool.capture.frame_size
        cap.intrinsics = self.g_pool.capture.intrinsics
        cap.timestamps = self.g_pool.capture.timestamps
        generator_args = (
            cap,
            gaze_data,
            np.deg2rad(self.max_dispersion),
            self.min_duration / 1000,
            self.max_duration / 1000,
            self.g_pool.min_data_confidence,
        )

        self.fixation_data = []
        self.fixation_start_ts = []
        self.fixation_stop_ts = []
        self.bg_task = bh.IPC_Logging_Task_Proxy(
            "Fixation detection", detect_fixations, args=generator_args
        )
        self.publish_empty()

    def recent_events(self, events):
        if self.bg_task:
            for progress, fixation_result in self.bg_task.fetch():
                self.status = progress
                if fixation_result:
                    serialized, start_ts, stop_ts = fixation_result
                    self.fixation_data.append(
                        fm.Serialized_Dict(msgpack_bytes=serialized)
                    )
                    self.fixation_start_ts.append(start_ts)
                    self.fixation_stop_ts.append(stop_ts)

                if self.fixation_data:
                    current_ts = self.fixation_stop_ts[-1]
                    progress = (current_ts - self.g_pool.timestamps[0]) / (
                        self.g_pool.timestamps[-1] - self.g_pool.timestamps[0]
                    )
                    self.menu_icon.indicator_stop = progress
            if self.bg_task.completed:
                self.status = f"{len(self.fixation_data)} fixations detected"
                self.correlate_and_publish_new()
                self.bg_task = None
                self.menu_icon.indicator_stop = 0.0

        frame = events.get("frame")
        if not frame:
            return

        self.last_frame_idx = frame.index
        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
        fixations = self.g_pool.fixations.by_ts_window(frame_window)
        events["fixations"] = fixations
        if self.show_fixations:
            for f in fixations:
                x = int(f["norm_pos"][0] * frame.width)
                y = int((1.0 - f["norm_pos"][1]) * frame.height)
                pm.transparent_circle(
                    frame.img,
                    (x, y),
                    radius=25.0,
                    color=(0.0, 1.0, 1.0, 1.0),
                    thickness=3,
                )
                cv2.putText(
                    frame.img,
                    "{}".format(f["id"]),
                    (x + 30, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 150, 100),
                )

        if self.current_fixation_details and self.prev_index != frame.index:
            info = ""
            for f in fixations:
                info += "Current fixation, {} of {}\n".format(
                    f["id"], len(self.g_pool.fixations)
                )
                info += "    Confidence: {:.2f}\n".format(f["confidence"])
                info += "    Duration: {:.2f} milliseconds\n".format(f["duration"])
                info += "    Dispersion: {:.3f} degrees\n".format(f["dispersion"])
                info += "    Frame range: {}-{}\n".format(
                    f["start_frame_index"] + 1, f["end_frame_index"] + 1
                )
                info += "    2d gaze pos: x={:.3f}, y={:.3f}\n".format(*f["norm_pos"])
                if "gaze_point_3d" in f:
                    info += "    3d gaze pos: x={:.3f}, y={:.3f}, z={:.3f}\n".format(
                        *f["gaze_point_3d"]
                    )
                else:
                    info += "    3d gaze pos: N/A\n"
                if f["id"] > 1:
                    prev_f = self.g_pool.fixations[f["id"] - 2]
                    time_lapsed = (
                        f["timestamp"] - prev_f["timestamp"] + prev_f["duration"] / 1000
                    )
                    info += "    Time since prev. fixation: {:.2f} seconds\n".format(
                        time_lapsed
                    )
                else:
                    info += "    Time since prev. fixation: N/A\n"

                if f["id"] < len(self.g_pool.fixations):
                    next_f = self.g_pool.fixations[f["id"]]
                    time_lapsed = (
                        next_f["timestamp"] - f["timestamp"] + f["duration"] / 1000
                    )
                    info += "    Time to next fixation: {:.2f} seconds\n".format(
                        time_lapsed
                    )
                else:
                    info += "    Time to next fixation: N/A\n"

            self.current_fixation_details.text = info
            self.prev_index = frame.index

    def correlate_and_publish_new(self):
        self.g_pool.fixations = pm.Affiliator(
            self.fixation_data, self.fixation_start_ts, self.fixation_stop_ts
        )
        self._fixations_changed_announcer.announce_new(
            delay=0.3,
            token_data=(
                self._gaze_changed_listener._current_token,
                self.max_dispersion,
                self.min_duration,
                self.max_duration,
                self.g_pool.min_data_confidence,
            ),
        )
        self.save_offline_data()

    def publish_empty(self):
        self.g_pool.fixations = pm.Affiliator([], [], [])
        self._fixations_changed_announcer.announce_new(token_data=())

    def correlate_and_publish_existing(self):
        self.g_pool.fixations = pm.Affiliator(
            self.fixation_data, self.fixation_start_ts, self.fixation_stop_ts
        )
        self._fixations_changed_announcer.announce_existing()

    def save_offline_data(self):
        with fm.PLData_Writer(self.data_dir, "fixations") as writer:
            for timestamp, datum in zip(self.fixation_start_ts, self.fixation_data):
                writer.append_serialized(timestamp, "fixation", datum.serialized)
        path_stop_ts = os.path.join(self.data_dir, "fixations_stop_timestamps.npy")
        np.save(path_stop_ts, self.fixation_stop_ts)
        path_meta = os.path.join(self.data_dir, "fixations.meta")
        fm.save_object(
            {
                "version": self.CACHE_VERSION,
                "config": self._cache_config(),
            },
            path_meta,
        )

    def load_offline_data(self):
        path_stop_ts = os.path.join(self.data_dir, "fixations_stop_timestamps.npy")
        fixation_stop_ts = np.load(path_stop_ts)
        path_meta = os.path.join(self.data_dir, "fixations.meta")
        meta = fm.load_object(path_meta)
        version_loaded = meta.get("version", -1)
        if version_loaded != self.CACHE_VERSION:
            raise self.VersionMismatchError(
                f"Expected version {self.CACHE_VERSION}, got {version_loaded}"
            )
        config_loaded = meta.get("config", None)
        config_expected = self._cache_config()
        if config_loaded != config_expected:
            raise self.ConfigMismatchError(
                f"Expected config {config_expected}, got {config_loaded}"
            )
        fixations = fm.load_pldata_file(self.data_dir, "fixations")
        if not (
            len(fixations.data) == len(fixations.timestamps) == len(fixation_stop_ts)
        ):
            raise self.DataMismatchError(
                f"Data inconsistent:\n"
                f"\tlen(fixations.data)={len(fixations.data)}\n"
                f"\tlen(fixations.timestamps)={len(fixations.timestamps)}\n"
                f"\tlen(fixation_stop_ts)={len(fixation_stop_ts)}"
            )
        self.fixation_data = fixations.data
        self.fixation_start_ts = fixations.timestamps
        self.fixation_stop_ts = fixation_stop_ts
        self.correlate_and_publish_existing()

    def _cache_config(self):
        return {
            "max_dispersion": self.max_dispersion,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "min_data_confidence": self.g_pool.min_data_confidence,
        }

    @classmethod
    def csv_representation_keys(self):
        return (
            "id",
            "start_timestamp",
            "duration",
            "start_frame_index",
            "end_frame_index",
            "norm_pos_x",
            "norm_pos_y",
            "dispersion",
            "confidence",
            "method",
            "gaze_point_3d_x",
            "gaze_point_3d_y",
            "gaze_point_3d_z",
            "base_data",
        )

    @classmethod
    def csv_representation_for_fixation(self, fixation):
        return (
            fixation["id"],
            fixation["timestamp"],
            fixation["duration"],
            fixation["start_frame_index"],
            fixation["end_frame_index"],
            fixation["norm_pos"][0],
            fixation["norm_pos"][1],
            fixation["dispersion"],
            fixation["confidence"],
            fixation["method"],
            *fixation.get(
                "gaze_point_3d", [None] * 3
            ),  # expanded, hence * at beginning
            " ".join(["{}".format(gp["timestamp"]) for gp in fixation["base_data"]]),
        )

    def export_fixations(self, export_window, export_dir):
        """
        between in and out mark

            fixation report:
                - fixation detection method and parameters
                - fixation count

            fixation list:
                id | start_timestamp | duration | start_frame_index | end_frame_index |
                norm_pos_x | norm_pos_y | dispersion | confidence | method |
                gaze_point_3d_x | gaze_point_3d_y | gaze_point_3d_z | base_data
        """
        if not self.fixation_data:
            logger.warning("No fixations in this recording nothing to export")
            return

        fixations_in_section = self.g_pool.fixations.by_ts_window(export_window)

        with open(
            os.path.join(export_dir, "fixations.csv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.csv_representation_keys())
            for f in fixations_in_section:
                csv_writer.writerow(self.csv_representation_for_fixation(f))
            logger.info("Created 'fixations.csv' file.")

        with open(
            os.path.join(export_dir, "fixation_report.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(("fixation classifier", "Dispersion_Duration"))
            csv_writer.writerow(("max_dispersion", f"{self.max_dispersion:0.3f} deg"))
            csv_writer.writerow(("min_duration", f"{self.min_duration:.0f} ms"))
            csv_writer.writerow(("max_duration", f"{self.max_duration:.0f} ms"))
            csv_writer.writerow("")
            csv_writer.writerow(("fixation_count", len(fixations_in_section)))
            logger.info("Created 'fixation_report.csv' file.")


class Fixation_Detector(Fixation_Detector_Base):
    """Dispersion-duration-based fixation detector.

    This plugin detects fixations based on a dispersion threshold in terms of
    degrees of visual angle with a minimal duration. It publishes the fixation
    as soon as it complies with the constraints (dispersion and duration). This
    might result in a series of overlapping fixations. These will have their id
    field set to the same value which can be used to merge overlapping fixations.

    If 3d pupil data is available the fixation dispersion will be calculated
    based on the positional angle of the eye. These fixations have their method
    field set to "pupil". If no 3d pupil data is available the plugin will
    assume that the gaze data is calibrated and calculate the dispersion in
    visual angle with in the coordinate system of the world camera. These
    fixations will have their method field set to "gaze".

    The Offline Fixation Detector yields fixations that do not overlap.
    """

    order = 0.19

    def __init__(self, g_pool, max_dispersion=3.0, min_duration=300, **kwargs):
        super().__init__(g_pool)
        self.history = []
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        self.id_counter = 0
        self.recent_fixation = None

    def recent_events(self, events):
        events["fixations"] = []
        gaze = events["gaze"]

        gaze = (
            gp for gp in gaze if gp["confidence"] >= self.g_pool.min_data_confidence
        )
        self.history.extend(gaze)
        self.history.sort(key=lambda gp: gp["timestamp"])

        if not self.history:
            self.recent_fixation = None
            return

        try:
            ts_oldest = self.history[0]["timestamp"]
            ts_newest = self.history[-1]["timestamp"]
            inconsistent_timestamps = ts_newest < ts_oldest
            if inconsistent_timestamps:
                self.reset_history()
                return

            age_threshold = ts_newest - self.min_duration / 1000.0
            # pop elements until only one element below the age threshold remains:
            while self.history[1]["timestamp"] < age_threshold:
                del self.history[0]  # remove outdated gaze points

        except IndexError:
            pass

        method = (
            FixationDetectionMethod.GAZE_3D
            if can_use_3d_gaze_mapping(self.history)
            else FixationDetectionMethod.GAZE_2D
        )
        base_data = self.history

        if len(base_data) <= 2 or (
            base_data[-1]["timestamp"] - base_data[0]["timestamp"]
            < self.min_duration / 1000.0
        ):
            self.recent_fixation = None
            return

        dispersion = gaze_dispersion(self.g_pool.capture, base_data, method)

        if dispersion < np.deg2rad(self.max_dispersion):
            new_fixation = fixation_from_data(dispersion, method, base_data)
            if self.recent_fixation:
                new_fixation["id"] = self.recent_fixation["id"]
            else:
                new_fixation["id"] = self.id_counter
                self.id_counter += 1

            self.replace_basedata_with_references(new_fixation)
            events["fixations"].append(new_fixation)
            self.recent_fixation = new_fixation
        else:
            self.recent_fixation = None

    def reset_history(self):
        logger.debug("Resetting history")
        self.history.clear()

    def replace_basedata_with_references(self, fixation):
        fixation["base_data"] = [
            (gaze["topic"], gaze["timestamp"]) for gaze in fixation["base_data"]
        ]

    def gl_display(self):
        if self.recent_fixation:
            fs = self.g_pool.capture.frame_size  # frame height
            pt = denormalize(self.recent_fixation["norm_pos"], fs, flip_y=True)
            draw_circle(
                pt, radius=48.0, stroke_width=10.0, color=RGBA(1.0, 1.0, 0.0, 1.0)
            )
            self.glfont.draw_text(pt[0] + 48.0, pt[1], str(self.recent_fixation["id"]))

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Fixation Detector"

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(ui.Info_Text(help_str))

        self.menu.append(
            ui.Slider(
                "max_dispersion",
                self,
                min=0.01,
                step=0.1,
                max=5.0,
                label="Maximum Dispersion [degrees]",
            )
        )
        self.menu.append(
            ui.Slider(
                "min_duration",
                self,
                min=10,
                step=10,
                max=4000,
                label="Minimum Duration [milliseconds]",
            )
        )

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

    def deinit_ui(self):
        self.remove_menu()
        self.glfont = None

    def get_init_dict(self):
        return {
            "max_dispersion": self.max_dispersion,
            "min_duration": self.min_duration,
        }
