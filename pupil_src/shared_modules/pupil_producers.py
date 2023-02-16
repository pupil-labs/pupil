"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import os
import typing as T
from contextlib import contextmanager
from itertools import chain

import data_changed
import file_methods as fm
import gl_utils
import numpy as np
import OpenGL.GL as gl
import player_methods as pm
import pyglui.cygl.utils as cygl_utils
import zmq
import zmq_tools
from observable import Observable
from plugin import System_Plugin_Base
from pupil_recording import PupilRecording, RecordingInfo
from pyglui import ui
from pyglui.pyfontstash import fontstash as fs
from video_capture.utils import VideoSet

logger = logging.getLogger(__name__)

COLOR_LEGEND_EYE_RIGHT = cygl_utils.RGBA(0.9844, 0.5938, 0.4023, 1.0)
COLOR_LEGEND_EYE_LEFT = cygl_utils.RGBA(0.668, 0.6133, 0.9453, 1.0)
NUMBER_SAMPLES_TIMELINE = 4000

DATA_KEY_CONFIDENCE = "confidence"
DATA_KEY_DIAMETER = "diameter_3d"


class Pupil_Producer_Base(Observable, System_Plugin_Base):
    uniqueness = "by_base_class"
    order = 0.01
    icon_chr = chr(0xEC12)
    icon_font = "pupil_icons"

    @classmethod
    @abc.abstractmethod
    def plugin_menu_label(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def pupil_data_source_selection_label(cls) -> str:
        return cls.plugin_menu_label()

    @staticmethod
    def available_pupil_producer_plugins(g_pool) -> list:
        def is_plugin_included(p, g_pool) -> bool:
            # Skip plugins that are not pupil producers
            if not issubclass(p, Pupil_Producer_Base):
                return False
            # Skip pupil producer stub
            if p is DisabledPupilProducer:
                return False
            # Skip pupil producers that are not available within g_pool context
            if not p.is_available_within_context(g_pool):
                return False
            return True

        return [
            p for p in g_pool.plugin_by_name.values() if is_plugin_included(p, g_pool)
        ]

    @classmethod
    def pupil_data_source_selection_order(cls) -> float:
        return float("inf")

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._pupil_changed_announcer = data_changed.Announcer(
            "pupil_positions", g_pool.rec_dir, plugin=self
        )
        self._pupil_changed_listener = data_changed.Listener(
            "pupil_positions", g_pool.rec_dir, plugin=self
        )
        self._pupil_changed_listener.add_observer(
            "on_data_changed", self._refresh_timelines
        )

    def init_ui(self):
        self.add_menu()

        pupil_producer_plugins = self.available_pupil_producer_plugins(self.g_pool)
        pupil_producer_plugins.sort(key=lambda p: p.pupil_data_source_selection_label())
        pupil_producer_plugins.sort(key=lambda p: p.pupil_data_source_selection_order())
        pupil_producer_labels = [
            p.pupil_data_source_selection_label() for p in pupil_producer_plugins
        ]

        self.menu.label = self.plugin_menu_label()
        self.menu_icon.order = 0.29
        self.menu_icon.tooltip = "Pupil Data"

        def open_plugin(p):
            self.notify_all({"subject": "start_plugin", "name": p.__name__})

        # We add the capture selection menu
        self.menu.append(
            ui.Selector(
                "pupil_producer",
                setter=open_plugin,
                getter=lambda: self.__class__,
                selection=pupil_producer_plugins,
                labels=pupil_producer_labels,
                label="Data Source",
            )
        )

        self.cache = {}
        self.cache_pupil_timeline_data(DATA_KEY_DIAMETER, detector_tag="3d")
        self.cache_pupil_timeline_data(
            DATA_KEY_CONFIDENCE,
            detector_tag="2d",
            ylim=(0.0, 1.0),
            fallback_detector_tag="3d",
        )

        self.glfont = fs.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_font("opensans")

        self.dia_timeline = ui.Timeline(
            label="Pupil Diameter 3D",
            draw_data_callback=self.draw_pupil_diameter,
            draw_label_callback=self.draw_dia_legend,
            content_height=40.0,
        )
        self.conf_timeline = ui.Timeline(
            "Pupil Confidence", self.draw_pupil_conf, self.draw_conf_legend
        )
        self.g_pool.user_timelines.append(self.dia_timeline)
        self.g_pool.user_timelines.append(self.conf_timeline)

    def _refresh_timelines(self):
        self.cache_pupil_timeline_data(DATA_KEY_DIAMETER, detector_tag="3d")
        self.cache_pupil_timeline_data(
            DATA_KEY_CONFIDENCE,
            detector_tag="2d",
            ylim=(0.0, 1.0),
            fallback_detector_tag="3d",
        )
        self.dia_timeline.refresh()
        self.conf_timeline.refresh()

    def deinit_ui(self):
        self.remove_menu()
        self.g_pool.user_timelines.remove(self.dia_timeline)
        self.g_pool.user_timelines.remove(self.conf_timeline)
        self.dia_timeline = None
        self.conf_timeline = None

    def recent_events(self, events):
        if "frame" in events:
            frm_idx = events["frame"].index
            window = pm.enclosing_window(self.g_pool.timestamps, frm_idx)
            events["pupil"] = self.g_pool.pupil_positions.by_ts_window(window)

    def cache_pupil_timeline_data(
        self,
        key: str,
        detector_tag: str,
        ylim=None,
        fallback_detector_tag: T.Optional[str] = None,
    ):
        world_start_stop_ts = [self.g_pool.timestamps[0], self.g_pool.timestamps[-1]]
        if not self.g_pool.pupil_positions:
            self.cache[key] = {
                "left": [],
                "right": [],
                "xlim": world_start_stop_ts,
                "ylim": [0, 1],
            }
        else:
            ts_data_pairs_right_left = [], []
            for eye_id in (0, 1):
                pupil_positions = self.g_pool.pupil_positions[eye_id, detector_tag]
                if not pupil_positions and fallback_detector_tag is not None:
                    pupil_positions = self.g_pool.pupil_positions[
                        eye_id, fallback_detector_tag
                    ]
                if pupil_positions:
                    t0, t1 = (
                        pupil_positions.timestamps[0],
                        pupil_positions.timestamps[-1],
                    )
                    timestamps_target = np.linspace(
                        t0, t1, NUMBER_SAMPLES_TIMELINE, dtype=np.float32
                    )

                    data_indeces = pm.find_closest(
                        pupil_positions.timestamps, timestamps_target
                    )
                    data_indeces = np.unique(data_indeces)
                    for idx in data_indeces:
                        ts_data_pair = (
                            pupil_positions.timestamps[idx],
                            pupil_positions[idx][key],
                        )
                        ts_data_pairs_right_left[eye_id].append(ts_data_pair)

            if ylim is None:
                # max_val must not be 0, else gl will crash
                all_pupil_data_chained = chain.from_iterable(ts_data_pairs_right_left)
                try:
                    # Outlier removal based on:
                    # https://en.wikipedia.org/wiki/Outlier#Tukey's_fences
                    min_val, max_val = np.quantile(
                        [pd[1] for pd in all_pupil_data_chained], [0.25, 0.75]
                    )
                    iqr = max_val - min_val
                    min_val -= 1.5 * iqr
                    max_val += 1.5 * iqr
                    ylim = min_val, max_val
                except IndexError:  # no pupil data available
                    ylim = 0.0, 1.0

            self.cache[key] = {
                "right": ts_data_pairs_right_left[0],
                "left": ts_data_pairs_right_left[1],
                "xlim": world_start_stop_ts,
                "ylim": ylim,
            }

    def draw_pupil_diameter(self, width, height, scale):
        self.draw_pupil_data(DATA_KEY_DIAMETER, width, height, scale)

    def draw_pupil_conf(self, width, height, scale):
        self.draw_pupil_data(DATA_KEY_CONFIDENCE, width, height, scale)

    def draw_pupil_data(self, key, width, height, scale):
        right = self.cache[key]["right"]
        left = self.cache[key]["left"]

        with gl_utils.Coord_System(*self.cache[key]["xlim"], *self.cache[key]["ylim"]):
            cygl_utils.draw_points(
                right, size=2.0 * scale, color=COLOR_LEGEND_EYE_RIGHT
            )
            cygl_utils.draw_points(left, size=2.0 * scale, color=COLOR_LEGEND_EYE_LEFT)

    def draw_dia_legend(self, width, height, scale):
        self.draw_legend(self.dia_timeline.label, width, height, scale)

        ylim = self.cache[DATA_KEY_DIAMETER]["ylim"]
        ylim_legend_text = f"Range: {ylim[0]:.1f} - {ylim[1]:.1f} mm"
        ylim_legend_pos = 26.0 * scale
        with self._legend_font(scale) as font:
            font.draw_text(width, ylim_legend_pos, ylim_legend_text)

    def draw_conf_legend(self, width, height, scale):
        self.draw_legend(self.conf_timeline.label, width, height, scale)

    def draw_legend(self, label, width, height, scale):
        legend_height = 13.0 * scale
        pad = 10 * scale

        with self._legend_font(scale) as font:
            font.draw_text(width, 0, label)
            font.draw_text(width / 2, legend_height, "left")
            font.draw_text(width, legend_height, "right")

        cygl_utils.draw_polyline(
            [(pad, 1.5 * legend_height), (width / 4, 1.5 * legend_height)],
            color=COLOR_LEGEND_EYE_LEFT,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )
        cygl_utils.draw_polyline(
            [
                (width / 2 + pad, 1.5 * legend_height),
                (width * 3 / 4, 1.5 * legend_height),
            ],
            color=COLOR_LEGEND_EYE_RIGHT,
            line_type=gl.GL_LINES,
            thickness=4.0 * scale,
        )

    @contextmanager
    def _legend_font(self, scale):
        self.glfont.push_state()
        try:
            self.glfont.set_align_string(v_align="right", h_align="top")
            self.glfont.set_size(15.0 * scale)
            yield self.glfont
        finally:
            self.glfont.pop_state()


class DisabledPupilProducer(Pupil_Producer_Base):
    """
    This is a stub implementation of a pupil producer,
    intended to be used when no (other) pupil producer is available.
    """

    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # Enable in Player only if Pupil Invisible recording
                return True
        return False

    @classmethod
    def plugin_menu_label(cls) -> str:
        raise RuntimeError()  # This method should never be called
        return "Disabled Pupil Producer"

    @classmethod
    def pupil_data_source_selection_order(cls) -> float:
        raise RuntimeError()  # This method should never be called
        return 0.1

    def __init__(self, g_pool):
        super().__init__(g_pool)
        # Create empty pupil_positions for all plugins that depend on it
        pupil_data = pm.PupilDataBisector(data=fm.PLData([], [], []))
        g_pool.pupil_positions = pupil_data
        self._pupil_changed_announcer.announce_existing()
        logger.debug("pupil positions changed")

    def init_ui(self):
        pass

    def deinit_ui(self):
        pass

    def _refresh_timelines(self):
        pass


class Pupil_From_Recording(Pupil_Producer_Base):
    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_MOBILE
            ):
                # Disable pupil from recording in Player if Pupil Mobile recording
                return False
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # Disable pupil from recording in Player if Pupil Invisible recording
                return False
        return super().is_available_within_context(g_pool)

    @classmethod
    def plugin_menu_label(cls) -> str:
        return "Pupil Data From Recording"

    @classmethod
    def pupil_data_source_selection_order(cls) -> float:
        return 1.0

    def __init__(self, g_pool):
        super().__init__(g_pool)

        pupil_data = pm.PupilDataBisector.load_from_file(g_pool.rec_dir, "pupil")
        g_pool.pupil_positions = pupil_data
        self._pupil_changed_announcer.announce_existing()
        logger.debug("pupil positions changed")

    def init_ui(self):
        super().init_ui()
        self.menu.append(ui.Info_Text("Using pupil data recorded by Pupil Capture."))


class Offline_Pupil_Detection(Pupil_Producer_Base):
    """docstring for Offline_Pupil_Detection"""

    session_data_version = 4
    session_data_name = "offline_pupil"

    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # Disable post-hoc pupil detector in Player if Pupil Invisible recording
                return False
        return super().is_available_within_context(g_pool)

    @classmethod
    def plugin_menu_label(cls) -> str:
        return "Post-Hoc Pupil Detection"

    @classmethod
    def pupil_data_source_selection_order(cls) -> float:
        return 2.0

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._detection_paused = False

        zmq_ctx = zmq.Context()
        self.data_sub = zmq_tools.Msg_Receiver(
            zmq_ctx,
            g_pool.ipc_sub_url,
            topics=("pupil", "notify.file_source"),
            hwm=100_000,
        )

        self.data_dir = os.path.join(g_pool.rec_dir, "offline_data")
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            session_meta_data = fm.load_object(
                os.path.join(self.data_dir, self.session_data_name + ".meta")
            )
            assert session_meta_data.get("version") == self.session_data_version
        except (AssertionError, FileNotFoundError):
            session_meta_data = {}
            session_meta_data["detection_status"] = ["unknown", "unknown"]

        self.detection_status = session_meta_data["detection_status"]

        self._pupil_data_store = pm.PupilDataCollector()
        pupil_data_from_cache = pm.PupilDataBisector.load_from_file(
            self.data_dir, self.session_data_name
        )
        self.publish_existing(pupil_data_from_cache)

        # Start offline pupil detection if not complete yet:
        self.eye_video_loc = [None, None]
        self.eye_frame_num = [0, 0]
        self.eye_frame_idx = [-1, -1]

        # start processes
        for eye_id in range(2):
            if self.detection_status[eye_id] != "complete":
                self.start_eye_process(eye_id)

    def start_eye_process(self, eye_id):
        potential_locs = [
            os.path.join(self.g_pool.rec_dir, f"eye{eye_id}{ext}")
            for ext in (".mjpeg", ".mp4", ".mkv")
        ]
        existing_locs = [loc for loc in potential_locs if os.path.exists(loc)]
        if not existing_locs:
            logger.error(f"no eye video for eye '{eye_id}' found.")
            self.detection_status[eye_id] = "No eye video found."
            return
        rec, file_ = os.path.split(existing_locs[0])
        set_name = os.path.splitext(file_)[0]
        self.videoset = VideoSet(rec, set_name, fill_gaps=False)
        self.videoset.load_or_build_lookup()
        if self.videoset.is_empty():
            logger.error(f"No videos for eye '{eye_id}' found.")
            self.detection_status[eye_id] = "No eye video found."
            return
        video_loc = existing_locs[0]
        n_valid_frames = np.count_nonzero(self.videoset.lookup.container_idx > -1)
        self.eye_frame_num[eye_id] = n_valid_frames
        self.eye_frame_idx = [-1, -1]

        capure_settings = "File_Source", {"source_path": video_loc, "timing": None}
        self.notify_all(
            {
                "subject": "eye_process.should_start",
                "eye_id": eye_id,
                "overwrite_cap_settings": capure_settings,
            }
        )
        self.eye_video_loc[eye_id] = video_loc
        self.detection_status[eye_id] = "Detecting..."

    @property
    def detection_progress(self) -> float:
        if not sum(self.eye_frame_num):
            return 0.0

        progress_by_eye = [0.0, 0.0]

        for eye_id in (0, 1):
            total_frames = self.eye_frame_num[eye_id]
            if total_frames > 0:
                current_index = self.eye_frame_idx[eye_id]
                progress = (current_index + 1) / total_frames
                progress = max(0.0, min(progress, 1.0))
            else:
                progress = 1.0
            progress_by_eye[eye_id] = progress

        return min(progress_by_eye)

    def stop_eye_process(self, eye_id):
        self.notify_all({"subject": "eye_process.should_stop", "eye_id": eye_id})
        self.eye_video_loc[eye_id] = None

    def recent_events(self, events):
        super().recent_events(events)
        while self.data_sub.new_data:
            topic = self.data_sub.recv_topic()
            remaining_frames = self.data_sub.recv_remaining_frames()
            if topic.startswith("pupil."):
                # pupil data only has one remaining frame
                payload_serialized = next(remaining_frames)
                pupil_datum = fm.Serialized_Dict(msgpack_bytes=payload_serialized)
                assert pm.PupilTopic.match(topic, eye_id=pupil_datum["id"])
                timestamp = pupil_datum["timestamp"]
                self._pupil_data_store.append(topic, pupil_datum, timestamp)
            else:
                payload = self.data_sub.deserialize_payload(*remaining_frames)
                if payload["subject"] == "file_source.video_finished":
                    for eye_id in (0, 1):
                        if self.eye_video_loc[eye_id] == payload["source_path"]:
                            logger.debug(f"eye {eye_id} process complete")
                            self.eye_frame_idx[eye_id] = self.eye_frame_num[eye_id]
                            self.detection_status[eye_id] = "complete"
                            self.stop_eye_process(eye_id)
                            break
                    if self.eye_video_loc == [None, None]:
                        data = self._pupil_data_store.as_pupil_data_bisector()
                        self.publish_new(pupil_data_bisector=data)
                if payload["subject"] == "file_source.current_frame_index":
                    for eye_id in (0, 1):
                        if self.eye_video_loc[eye_id] == payload["source_path"]:
                            self.eye_frame_idx[eye_id] = payload["index"]

        self.menu_icon.indicator_stop = self.detection_progress

    def publish_existing(self, pupil_data_bisector):
        self.g_pool.pupil_positions = pupil_data_bisector
        self._pupil_changed_announcer.announce_existing()

    def publish_new(self, pupil_data_bisector):
        self.g_pool.pupil_positions = pupil_data_bisector
        self._pupil_changed_announcer.announce_new()
        logger.debug("pupil positions changed")
        self.save_offline_data()

    def on_notify(self, notification):
        super().on_notify(notification)
        if notification["subject"] == "eye_process.started":
            pass
        elif notification["subject"] == "eye_process.stopped":
            self.eye_video_loc[notification["eye_id"]] = None

    def cleanup(self):
        self.stop_eye_process(0)
        self.stop_eye_process(1)
        # close sockets before context is terminated
        self.data_sub = None
        self.save_offline_data()

    def save_offline_data(self):
        self.g_pool.pupil_positions.save_to_file(self.data_dir, "offline_pupil")
        session_data = {}
        session_data["detection_status"] = self.detection_status
        session_data["version"] = self.session_data_version
        cache_path = os.path.join(self.data_dir, "offline_pupil.meta")
        fm.save_object(session_data, cache_path)
        logger.info(f"Cached detected pupil data to {cache_path}")

    def redetect(self):
        self._pupil_data_store.clear()
        self.g_pool.pupil_positions = self._pupil_data_store.as_pupil_data_bisector()
        self._pupil_changed_announcer.announce_new()
        self.detection_finished_flag = False
        self.detection_paused = False
        for eye_id in range(2):
            if self.eye_video_loc[eye_id] is None:
                self.start_eye_process(eye_id)
            else:
                self.notify_all(
                    {
                        "subject": "file_source.seek",
                        "frame_index": 0,
                        "source_path": self.eye_video_loc[eye_id],
                    }
                )

    def init_ui(self):
        super().init_ui()
        self.menu.append(ui.Info_Text("Detect pupil positions from eye videos."))
        self.menu.append(ui.Switch("detection_paused", self, label="Pause detection"))
        self.menu.append(ui.Button("Redetect", self.redetect))
        self.menu.append(
            ui.Text_Input(
                "0",
                label="eye0:",
                getter=lambda: self.detection_status[0],
                setter=lambda _: _,
            )
        )
        self.menu.append(
            ui.Text_Input(
                "1",
                label="eye1:",
                getter=lambda: self.detection_status[1],
                setter=lambda _: _,
            )
        )

        progress_slider = ui.Slider(
            "detection_progress",
            label="Detection Progress",
            getter=lambda: 100 * self.detection_progress,
            setter=lambda _: _,
        )
        progress_slider.display_format = "%3.0f%%"
        self.menu.append(progress_slider)

    @property
    def detection_paused(self):
        return self._detection_paused

    @detection_paused.setter
    def detection_paused(self, should_pause):
        self._detection_paused = should_pause
        for eye_id in range(2):
            if self.eye_video_loc[eye_id] is not None:
                subject = "file_source." + (
                    "should_pause" if should_pause else "should_play"
                )
                self.notify_all(
                    {"subject": subject, "source_path": self.eye_video_loc[eye_id]}
                )
