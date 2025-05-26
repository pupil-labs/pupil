"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import csv
import logging
import os
import time
from collections import deque

import csv_utils
import data_changed
import file_methods as fm
import gl_utils
import numpy as np
import OpenGL.GL as gl
import player_methods as pm
import pyglui.cygl.utils as cygl_utils
from observable import Observable
from plugin import Plugin
from pupil_recording import PupilRecording, RecordingInfo
from pyglui import ui
from pyglui.pyfontstash import fontstash as fs
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


activity_color = cygl_utils.RGBA(0.6602, 0.8594, 0.4609, 0.8)
blink_color = cygl_utils.RGBA(0.9961, 0.3789, 0.5313, 0.8)
threshold_color = cygl_utils.RGBA(0.9961, 0.8438, 0.3984, 0.8)


class Blink_Detection(Plugin):
    """
    This plugin implements a blink detection algorithm, based on sudden drops in the
    pupil detection confidence.
    """

    order = 0.8
    icon_chr = chr(0xE81A)
    icon_font = "pupil_icons"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Blink Detector"

    def __init__(
        self,
        g_pool,
        history_length=0.2,
        onset_confidence_threshold=0.5,
        offset_confidence_threshold=0.5,
    ):
        super().__init__(g_pool)
        self.history_length = history_length  # unit: seconds
        self.onset_confidence_threshold = onset_confidence_threshold
        self.offset_confidence_threshold = offset_confidence_threshold

        self.history = deque()
        self.menu = None
        self._recent_blink = None

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Blink Detector"
        self.menu.append(
            ui.Info_Text(
                "This plugin detects blink onsets and "
                "offsets based on confidence drops."
            )
        )
        self.menu.append(
            ui.Slider(
                "history_length",
                self,
                label="Filter length [seconds]",
                min=0.1,
                max=0.5,
                step=0.05,
            )
        )
        self.menu.append(
            ui.Slider(
                "onset_confidence_threshold",
                self,
                label="Onset confidence threshold",
                min=0.0,
                max=1.0,
                step=0.05,
            )
        )
        self.menu.append(
            ui.Slider(
                "offset_confidence_threshold",
                self,
                label="Offset confidence threshold",
                min=0.0,
                max=1.0,
                step=0.05,
            )
        )

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events={}):
        events["blinks"] = []
        self._recent_blink = None

        pupil = events.get("pupil", [])
        pupil = filter(lambda p: "2d" in p["topic"], pupil)
        self.history.extend(pupil)

        try:
            ts_oldest = self.history[0]["timestamp"]
            ts_newest = self.history[-1]["timestamp"]
            inconsistent_timestamps = ts_newest < ts_oldest
            if inconsistent_timestamps:
                self.reset_history()
                return

            age_threshold = ts_newest - self.history_length
            # pop elements until only one element below the age threshold remains:
            while self.history[1]["timestamp"] < age_threshold:
                self.history.popleft()  # remove outdated gaze points
        except IndexError:
            pass

        filter_size = len(self.history)
        if filter_size < 2 or ts_newest - ts_oldest < self.history_length:
            return

        activity = np.fromiter((pp["confidence"] for pp in self.history), dtype=float)
        blink_filter = np.ones(filter_size) / filter_size
        blink_filter[filter_size // 2 :] *= -1

        if filter_size % 2 == 1:  # make filter symmetrical
            blink_filter[filter_size // 2] = 0.0

        # The theoretical response maximum is +-0.5
        # Response of +-0.45 seems sufficient for a confidence of 1.
        filter_response = activity @ blink_filter / 0.45

        if (
            -self.offset_confidence_threshold
            <= filter_response
            <= self.onset_confidence_threshold
        ):
            return  # response cannot be classified as blink onset or offset
        elif filter_response > self.onset_confidence_threshold:
            blink_type = "onset"
        else:
            blink_type = "offset"

        confidence = min(abs(filter_response), 1.0)  # clamp conf. value at 1.
        # Add info to events
        blink_entry = {
            "topic": "blinks",
            "type": blink_type,
            "confidence": confidence,
            "base_data": list(self.history),
            "timestamp": self.history[len(self.history) // 2]["timestamp"],
            "record": True,
        }
        events["blinks"].append(blink_entry)
        self._recent_blink = blink_entry

    def reset_history(self):
        logger.debug("Resetting history")
        self.history.clear()

    def get_init_dict(self):
        return {
            "history_length": self.history_length,
            "onset_confidence_threshold": self.onset_confidence_threshold,
            "offset_confidence_threshold": self.offset_confidence_threshold,
        }


class Offline_Blink_Detection(Observable, Blink_Detection):
    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # Disable blink detector in Player if Pupil Invisible recording
                return False
        return super().is_available_within_context(g_pool)

    def __init__(
        self,
        g_pool,
        history_length=0.2,
        onset_confidence_threshold=0.5,
        offset_confidence_threshold=0.5,
    ):
        self._history_length = None
        self._onset_confidence_threshold = None
        self._offset_confidence_threshold = None

        super().__init__(
            g_pool,
            history_length,
            onset_confidence_threshold,
            offset_confidence_threshold,
        )
        self.filter_response = []
        self.response_classification = []
        self.timestamps = []
        g_pool.blinks = pm.Affiliator()
        self.cache = {"response_points": (), "class_points": (), "thresholds": ()}

        self.pupil_positions_listener = data_changed.Listener(
            "pupil_positions", g_pool.rec_dir, plugin=self
        )
        self.pupil_positions_listener.add_observer(
            "on_data_changed", self._on_pupil_positions_changed
        )

    def _pupil_data(self):
        data = self.g_pool.pupil_positions[..., "2d"]
        if not data:
            # Fall back to 3d data
            data = self.g_pool.pupil_positions[..., "3d"]
        return data

    def init_ui(self):
        super().init_ui()
        self.glfont = fs.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_font("opensans")
        self.timeline = ui.Timeline(
            "Blink Detection", self.draw_activation, self.draw_legend
        )
        self.timeline.content_height *= 2
        self.g_pool.user_timelines.append(self.timeline)

    def deinit_ui(self):
        super().deinit_ui()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None

    def recent_events(self, events):
        pass

    def gl_display(self):
        pass

    def on_notify(self, notification):
        if notification["subject"] == "blink_detection.should_recalculate":
            self.recalculate()
        elif notification["subject"] == "blinks_changed":
            self.cache_activation()
            try:
                self.timeline.refresh()
            except AttributeError:
                pass
        elif notification["subject"] == "should_export":
            self.export(notification["ts_window"], notification["export_dir"])

    def _on_pupil_positions_changed(self):
        logger.info("Pupil postions changed. Recalculating.")
        self.recalculate()

    def export(self, export_window, export_dir):
        """
        Between in and out mark

            blink_detection_report.csv:
                - history lenght
                - onset threshold
                - offset threshold

            blinks.csv:
                id | start_timestamp | duration | end_timestamp |
                start_frame_index | index | end_frame_index |
                confidence | filter_response | base_data
        """
        if not self.g_pool.blinks:
            logger.warning(
                "No blinks were detected in this recording. Nothing to export."
            )
            return

        header = (
            "id",
            "start_timestamp",
            "duration",
            "end_timestamp",
            "start_frame_index",
            "index",
            "end_frame_index",
            "confidence",
            "filter_response",
            "base_data",
        )

        blinks_in_section = self.g_pool.blinks.by_ts_window(export_window)

        with open(
            os.path.join(export_dir, "blinks.csv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
            for b in blinks_in_section:
                csv_writer.writerow(self.csv_representation_for_blink(b, header))
            logger.info("Created 'blinks.csv' file.")

        with open(
            os.path.join(export_dir, "blink_detection_report.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            csv_utils.write_key_value_file(
                csvfile,
                {
                    "history_length": self.history_length,
                    "onset_confidence_threshold": self.onset_confidence_threshold,
                    "offset_confidence_threshold": self.offset_confidence_threshold,
                    "blinks_exported": len(blinks_in_section),
                },
            )
            logger.info("Created 'blink_detection_report.csv' file.")

    def recalculate(self):
        t0 = time.perf_counter()
        all_pp = self._pupil_data()
        if not all_pp:
            self.filter_response = []
            self.response_classification = []
            self.timestamps = []
            self.consolidate_classifications()
            return

        self.timestamps = all_pp.timestamps
        total_time = self.timestamps[-1] - self.timestamps[0]

        conf_iter = (pp["confidence"] for pp in all_pp)
        activity = np.fromiter(conf_iter, dtype=float, count=len(all_pp))
        total_time = all_pp[-1]["timestamp"] - all_pp[0]["timestamp"]
        filter_size = 2 * round(len(all_pp) * self.history_length / total_time / 2.0)
        blink_filter = np.ones(filter_size) / filter_size

        # This is different from the online filter. Convolution will flip
        # the filter and result in a reverse filter response. Therefore
        # we set the first half of the filter to -1 instead of the second
        # half such that we get the expected result.
        blink_filter[: filter_size // 2] *= -1

        # The theoretical response maximum is +-0.5
        # Response of +-0.45 seems sufficient for a confidence of 1.
        self.filter_response = fftconvolve(activity, blink_filter, "same") / 0.45

        onsets = self.filter_response > self.onset_confidence_threshold
        offsets = self.filter_response < -self.offset_confidence_threshold

        self.response_classification = np.zeros(self.filter_response.shape)
        self.response_classification[onsets] = 1.0
        self.response_classification[offsets] = -1.0

        self.consolidate_classifications()

        tm1 = time.perf_counter()
        logger.debug(
            "Recalculating took\n\t{:.4f}sec for {} pp\n\t{} pp/sec\n\tsize: {}".format(
                tm1 - t0, len(all_pp), len(all_pp) / (tm1 - t0), filter_size
            )
        )

    def consolidate_classifications(self):
        blink = None
        state = "no blink"  # others: 'blink started' | 'blink ending'
        blink_data = deque()
        blink_start_ts = deque()
        blink_stop_ts = deque()
        counter = 1

        # NOTE: Cache result for performance reasons
        pupil_data = self._pupil_data()

        def start_blink(idx):
            nonlocal blink
            nonlocal state
            nonlocal counter
            blink = {
                "topic": "blink",
                "__start_response_index__": idx,
                "start_timestamp": self.timestamps[idx],
                "id": counter,
            }
            state = "blink started"
            counter += 1

        def blink_finished(idx):
            nonlocal blink

            # get tmp pupil idx
            start_idx = blink["__start_response_index__"]
            del blink["__start_response_index__"]

            blink["end_timestamp"] = self.timestamps[idx]
            blink["timestamp"] = (blink["end_timestamp"] + blink["start_timestamp"]) / 2
            blink["duration"] = blink["end_timestamp"] - blink["start_timestamp"]
            blink["base_data"] = pupil_data[start_idx:idx].tolist()
            blink["filter_response"] = self.filter_response[start_idx:idx].tolist()
            # blink confidence is the mean of the absolute filter response
            # during the blink event, clamped at 1.
            blink["confidence"] = min(
                float(np.abs(blink["filter_response"]).mean()), 1.0
            )

            # correlate world indices
            ts_start, ts_end = blink["start_timestamp"], blink["end_timestamp"]

            idx_start, idx_end = np.searchsorted(
                self.g_pool.timestamps, [ts_start, ts_end]
            )
            # fix `list index out of range` error
            idx_end = min(idx_end, len(self.g_pool.timestamps) - 1)
            blink["start_frame_index"] = int(idx_start)
            blink["end_frame_index"] = int(idx_end)
            blink["index"] = int((idx_start + idx_end) // 2)

            blink_data.append(fm.Serialized_Dict(python_dict=blink))
            blink_start_ts.append(ts_start)
            blink_stop_ts.append(ts_end)

        for idx, classification in enumerate(self.response_classification):
            if state == "no blink" and classification > 0:
                start_blink(idx)
            elif state == "blink started" and classification == -1:
                state = "blink ending"
            elif state == "blink ending" and classification >= 0:
                blink_finished(idx - 1)  # blink ended previously
                if classification > 0:
                    start_blink(0)
                else:
                    blink = None
                    state = "no blink"

        if state == "blink ending":
            # only finish blink if it was already ending
            blink_finished(idx)  # idx is the last possible idx

        self.g_pool.blinks = pm.Affiliator(blink_data, blink_start_ts, blink_stop_ts)
        self.notify_all({"subject": "blinks_changed", "delay": 0.2})

    def cache_activation(self):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]
        self.cache["thresholds"] = (
            (t0, self.onset_confidence_threshold),
            (t1, self.onset_confidence_threshold),
            (t0, -self.offset_confidence_threshold),
            (t1, -self.offset_confidence_threshold),
        )

        self.cache["response_points"] = tuple(
            zip(self.timestamps, self.filter_response)
        )
        if len(self.cache["response_points"]) == 0:
            self.cache["class_points"] = ()
            return

        class_points = deque([(t0, -0.9)])
        for b in self.g_pool.blinks:
            class_points.append((b["start_timestamp"], -0.9))
            class_points.append((b["start_timestamp"], 0.9))
            class_points.append((b["end_timestamp"], 0.9))
            class_points.append((b["end_timestamp"], -0.9))
        class_points.append((t1, -0.9))
        self.cache["class_points"] = tuple(class_points)

    def draw_activation(self, width, height, scale):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]
        with gl_utils.Coord_System(t0, t1, -1, 1):
            cygl_utils.draw_polyline(
                self.cache["response_points"],
                color=activity_color,
                line_type=gl.GL_LINE_STRIP,
                thickness=scale,
            )
            cygl_utils.draw_polyline(
                self.cache["class_points"],
                color=blink_color,
                line_type=gl.GL_LINE_STRIP,
                thickness=scale,
            )
            cygl_utils.draw_polyline(
                self.cache["thresholds"],
                color=threshold_color,
                line_type=gl.GL_LINES,
                thickness=scale,
            )

    def draw_legend(self, width, height, scale):
        self.glfont.push_state()
        self.glfont.set_align_string(v_align="right", h_align="top")
        self.glfont.set_size(15.0 * scale)
        self.glfont.draw_text(width, 0, self.timeline.label)

        legend_height = 13.0 * scale
        pad = 10 * scale

        self.glfont.draw_text(width, legend_height, "Activity")
        cygl_utils.draw_polyline(
            [
                (pad, legend_height + pad * 2 / 3),
                (width / 2, legend_height + pad * 2 / 3),
            ],
            color=activity_color,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )
        legend_height += 1.5 * pad

        self.glfont.draw_text(width, legend_height, "Thresholds")
        cygl_utils.draw_polyline(
            [
                (pad, legend_height + pad * 2 / 3),
                (width / 2, legend_height + pad * 2 / 3),
            ],
            color=threshold_color,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )
        legend_height += 1.5 * pad

        self.glfont.draw_text(width, legend_height, "Blinks")
        cygl_utils.draw_polyline(
            [
                (pad, legend_height + pad * 2 / 3),
                (width / 2, legend_height + pad * 2 / 3),
            ],
            color=blink_color,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )

    @property
    def history_length(self):
        return self._history_length

    @history_length.setter
    def history_length(self, val):
        if self._history_length != val:
            self.notify_all(
                {"subject": "blink_detection.should_recalculate", "delay": 0.2}
            )
        self._history_length = val

    @property
    def onset_confidence_threshold(self):
        return self._onset_confidence_threshold

    @onset_confidence_threshold.setter
    def onset_confidence_threshold(self, val):
        if self._onset_confidence_threshold != val:
            self.notify_all(
                {"subject": "blink_detection.should_recalculate", "delay": 0.2}
            )
        self._onset_confidence_threshold = val

    @property
    def offset_confidence_threshold(self):
        return self._offset_confidence_threshold

    @offset_confidence_threshold.setter
    def offset_confidence_threshold(self, val):
        if self._offset_confidence_threshold != val:
            self.notify_all(
                {"subject": "blink_detection.should_recalculate", "delay": 0.2}
            )
        self._offset_confidence_threshold = val

    def csv_representation_for_blink(self, b, header):
        data = [b[k] for k in header if k not in ("filter_response", "base_data")]
        try:
            resp = " ".join([f"{val}" for val in b["filter_response"]])
            data.insert(header.index("filter_response"), resp)
        except IndexError:
            pass
        try:
            base = " ".join(["{}".format(pp["timestamp"]) for pp in b["base_data"]])
            data.insert(header.index("base_data"), base)
        except IndexError:
            pass
        return data
