"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


# stdlib
import os
import abc
import csv
import enum
import typing
import logging
import traceback
import operator
import functools
import itertools
import collections
import bisect
from typing import _Protocol as Protocol


# local
from tasklib import interface
from tasklib.background.task import BackgroundGeneratorFunction
from tasklib.background.patches import Patch, IPCLoggingPatch
from tasklib.manager import PluginTaskManager
import file_methods as fm
import player_methods as pm

import methods
from plugin import Analysis_Plugin_Base
from observable import Observable
import video_capture as vc

# third-party
import nslr_hmm
import numpy as np
import cv2
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle
from pyglui.pyfontstash import fontstash



class Notification_Subject:
    SHOULD_RECALCULATE = "segmentation_detector.should_recalculate"
    SEGMENTATION_CHANGED = "segmentation_changed"


class _Seek_Notification_Subject:
    SHOULD_SEEK = "seek_control.should_seek"


class _Eye_Movement_Detector_Base(Analysis_Plugin_Base):
    icon_chr = chr(0xEC03)
    icon_font = "pupil_icons"


class Offline_Eye_Movement_Detector(Observable, _Eye_Movement_Detector_Base):
    """Eye movement classification detector based on segmented linear regression.
    """

    MENU_LABEL_TEXT = "Eye Movement Detector"

    def __init__(self, g_pool, show_segmentation=True):
        super().__init__(g_pool)
        self.show_segmentation = show_segmentation
        self.current_segment_index = None
        self.current_segment_details = None
        self.eye_movement_detection_yields = deque()
        self.status = ""

        self.task_manager = PluginTaskManager(self)
        self.eye_movement_task = None

        self.notify_all(
            {"subject": Notification_Subject.SHOULD_RECALCULATE, "delay": 0.5}
        )

    def init_ui(self):
        self.add_menu()
        self.menu.label = type(self).MENU_LABEL_TEXT

        def jump_next_segment(_):
            if len(self.g_pool.eye_movement_segments) < 1:
                logger.warning("No eye movement segments availabe")
                return

            # Set current segment index to next one, or to 0 if not available
            self.current_segment_index = (
                self.current_segment_index if self.current_segment_index else 0
            )
            self.current_segment_index = (self.current_segment_index + 1) % len(
                self.g_pool.eye_movement_segments
            )

            next_segment_ts = self.g_pool.eye_movement_segments[
                self.current_segment_index
            ].start_frame_timestamp

            self.notify_all(
                {
                    "subject": _Seek_Notification_Subject.SHOULD_SEEK,
                    "timestamp": next_segment_ts,
                }
            )

        def jump_prev_segment(_):
            if len(self.g_pool.eye_movement_segments) < 1:
                logger.warning("No segmentation availabe")
                return

            # Set current segment index to previous one, or to 0 if not available
            self.current_segment_index = (
                self.current_segment_index if self.current_segment_index else 0
            )
            self.current_segment_index = (self.current_segment_index - 1) % len(
                self.g_pool.eye_movement_segments
            )

            prev_segment_ts = self.g_pool.eye_movement_segments[
                self.current_segment_index
            ].start_frame_timestamp

            self.notify_all(
                {
                    "subject": _Seek_Notification_Subject.SHOULD_SEEK,
                    "timestamp": prev_segment_ts,
                }
            )

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(ui.Info_Text(help_str))

        self.menu.append(
            ui.Info_Text("Press the export button or type 'e' to start the export.")
        )

        detection_status_input = ui.Text_Input(
            "status", self, label="Detection progress:", setter=lambda x: None
        )

        show_segmentation_switch = ui.Switch(
            "show_segmentation", self, label="Show segmentation"
        )

        self.current_segment_details = ui.Info_Text("")

        self.next_segment_button = ui.Thumb(
            "jump_next_segment",
            setter=jump_next_segment,
            getter=lambda: False,
            label=chr(0xE044),
            hotkey="f",
            label_font="pupil_icons",
        )
        self.next_segment_button.status_text = "Next Segment"

        self.prev_segment_button = ui.Thumb(
            "jump_prev_segment",
            setter=jump_prev_segment,
            getter=lambda: False,
            label=chr(0xE045),
            hotkey="F",
            label_font="pupil_icons",
        )
        self.prev_segment_button.status_text = "Previous Segment"

        self.menu.append(detection_status_input)
        self.menu.append(show_segmentation_switch)
        self.menu.append(self.current_segment_details)

        self.g_pool.quickbar.append(self.next_segment_button)
        self.g_pool.quickbar.append(self.prev_segment_button)

    def deinit_ui(self):
        self.remove_menu()
        self.g_pool.quickbar.remove(self.next_segment_button)
        self.g_pool.quickbar.remove(self.prev_segment_button)
        self.current_segment_details = None
        self.next_segment_button = None
        self.prev_segment_button = None

    def get_init_dict(self):
        return {"show_segmentation": self.show_segmentation}

    def on_notify(self, notification):
        if notification["subject"] == "gaze_positions_changed":
            logger.info("Gaze postions changed. Recalculating.")
            self._classify()
        elif notification["subject"] == Notification_Subject.SHOULD_RECALCULATE:
            self._classify()
        elif notification["subject"] == "should_export":
            self.export_eye_movement(notification["range"], notification["export_dir"])

    def _classify(self):
        """
        classify eye movement
        """

        if self.g_pool.app == "exporter":
            return

        if self.eye_movement_task and self.eye_movement_task.running:
            self.eye_movement_task.kill(grace_period=1)

        capture = Immutable_Capture(self.g_pool.capture)
        gaze_data: Gaze_Data = [gp.serialized for gp in self.g_pool.gaze_positions]

        self.eye_movement_task = Eye_Movement_Detection_Task(args=(capture, gaze_data))
        self.task_manager.add_task(self.eye_movement_task)

        self.eye_movement_task.add_observers(
            on_started=self.on_task_started,
            on_yield=self.on_task_yield,
            on_completed=self.on_task_completed,
            on_ended=self.on_task_ended,
            on_exception=self.on_task_exception,
            on_canceled_or_killed=self.on_task_canceled_or_killed,
        )
        self.eye_movement_task.start()

    def on_task_started(self):
        self.eye_movement_detection_yields = collections.deque()

    def on_task_yield(self, yield_value):

        status, serialized = yield_value
        self.status = status

        if serialized:
            segment = Classified_Segment.from_msgpack(serialized)
            self.eye_movement_detection_yields.append(segment)

            current_ts = segment.end_frame_timestamp
            total_start_ts = self.g_pool.timestamps[0]
            total_end_ts = self.g_pool.timestamps[-1]

            current_duration = current_ts - total_start_ts
            total_duration = total_end_ts - total_start_ts

            progress = min(0.0, max(current_duration / total_duration, 1.0))
            self.menu_icon.indicator_stop = progress

    def on_task_exception(self, exception):
        pass

    def on_task_completed(self, return_value):
        self.status = "{} segments detected".format(
            len(self.eye_movement_detection_yields)
        )
        self.correlate_and_publish()

    def on_task_canceled_or_killed(self):
        pass

    def on_task_ended(self):
        if self.menu_icon:
            self.menu_icon.indicator_stop = 0.0

    def _ui_draw_visible_segments(self, frame, visible_segments):
        if not self.show_segmentation:
            return
        for segment in visible_segments:
            segment.draw_on_frame(frame)

    def _ui_update_segment_detail_text(self, index, total_count, focused_segment):

        if (index is None) or (total_count < 1) or (not focused_segment):
            self.current_segment_details.text = ""
            return

        info = ""
        prev_segment = (
            self.g_pool.eye_movement_segments[index - 1] if index > 0 else None
        )
        next_segment = (
            self.g_pool.eye_movement_segments[self.current_segment_index + 1]
            if self.current_segment_index < len(self.g_pool.eye_movement_segments) - 1
            else None
        )

        info += "Current segment, {} of {}\n".format(index + 1, total_count)
        info += "    ID: {}\n".format(focused_segment.id)
        info += "    Classification: {}\n".format(focused_segment.segment_class.value)
        info += "    Confidence: {:.2f}\n".format(focused_segment.confidence)
        info += "    Duration: {:.2f} milliseconds\n".format(focused_segment.duration)
        info += "    Frame range: {}-{}\n".format(
            focused_segment.start_frame_index + 1, focused_segment.end_frame_index + 1
        )
        info += "    2d gaze pos: x={:.3f}, y={:.3f}\n".format(
            *focused_segment.norm_pos
        )
        if focused_segment.gaze_point_3d:
            info += "    3d gaze pos: x={:.3f}, y={:.3f}, z={:.3f}\n".format(
                *focused_segment.gaze_point_3d
            )
        else:
            info += "    3d gaze pos: N/A\n"

        if prev_segment:
            info += "    Time since prev. segment: {:.2f} seconds\n".format(
                prev_segment.duration / 1000
            )
        else:
            info += "    Time since prev. segment: N/A\n"

        if next_segment:
            info += "    Time to next segment: {:.2f} seconds\n".format(
                focused_segment.duration / 1000
            )
        else:
            info += "    Time to next segment: N/A\n"
        self.current_segment_details.text = info

    def recent_events(self, events):

        frame = events.get("frame")
        if not frame:
            return

        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
        visible_segments: typing.Iterable[
            Classified_Segment
        ] = self.g_pool.eye_movement_segments.by_ts_window(frame_window)
        events[EYE_MOVEMENT_EVENT_KEY] = visible_segments

        def _find_focused_segment(visible_segments):
            current_segment = None
            visible_segments = visible_segments if visible_segments else []
            current_segment_index = self.current_segment_index

            if current_segment_index:
                current_segment_index = current_segment_index % len(
                    self.g_pool.eye_movement_segments
                )
                current_segment = self.g_pool.eye_movement_segments[
                    current_segment_index
                ]

            if not visible_segments:
                return current_segment_index, current_segment

            if (not current_segment in visible_segments) and len(visible_segments) > 0:
                current_segment = visible_segments[0]
                current_segment_index = self.g_pool.eye_movement_segments.data.index(
                    current_segment
                )

            return current_segment_index, current_segment

        self.current_segment_index, current_segment = _find_focused_segment(
            visible_segments
        )

        self._ui_draw_visible_segments(frame, visible_segments)
        self._ui_update_segment_detail_text(
            self.current_segment_index,
            len(self.g_pool.eye_movement_segments),
            current_segment,
        )

    def correlate_and_publish(self):
        self.g_pool.eye_movement_segments = pm.Affiliator(
            self.eye_movement_detection_yields,
            [
                segment.start_frame_timestamp
                for segment in self.eye_movement_detection_yields
            ],
            [
                segment.end_frame_timestamp
                for segment in self.eye_movement_detection_yields
            ],
        )
        self.notify_all(
            {"subject": Notification_Subject.SEGMENTATION_CHANGED, "delay": 1}
        )

    @classmethod
    def csv_schema(cls):
        return [
            ("id", lambda seg: seg.id),
            ("method", lambda seg: seg.method.value),
            ("segment_class", lambda seg: seg.segment_class.value),
            ("start_frame_index", lambda seg: seg.start_frame_index),
            ("end_frame_index", lambda seg: seg.end_frame_index),
            ("start_timestamp", lambda seg: seg.start_frame_timestamp),
            ("end_timestamp", lambda seg: seg.end_frame_timestamp),
            ("duration", lambda seg: seg.duration),
            ("confidence", lambda seg: seg.confidence),
            ("norm_pos_x", lambda seg: seg.norm_pos[0]),
            ("norm_pos_y", lambda seg: seg.norm_pos[1]),
            (
                "gaze_point_3d_x",
                lambda seg: seg.gaze_point_3d[0] if seg.gaze_point_3d else "",
            ),
            (
                "gaze_point_3d_y",
                lambda seg: seg.gaze_point_3d[1] if seg.gaze_point_3d else "",
            ),
            (
                "gaze_point_3d_z",
                lambda seg: seg.gaze_point_3d[2] if seg.gaze_point_3d else "",
            ),
        ]

    @classmethod
    def csv_header(cls):
        return tuple(label for label, _ in cls.csv_schema())

    @classmethod
    def csv_row(cls, segment):
        return tuple(str(getter(segment)) for _, getter in cls.csv_schema())

    def export_eye_movement(self, export_range, export_dir):

        if not self.eye_movement_detection_yields:
            logger.warning("No fixations in this recording nothing to export")
            return

        export_window = pm.exact_window(self.g_pool.timestamps, export_range)
        segments_in_section = self.g_pool.eye_movement_segments.by_ts_window(
            export_window
        )

        segment_export_filename = "eye_movement.csv"
        segment_export_full_path = os.path.join(export_dir, segment_export_filename)

        with open(
            segment_export_full_path, "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(type(self).csv_header())
            for segment in segments_in_section:
                csv_writer.writerow(type(self).csv_row(segment))
            logger.info("Created '{}' file.".format(segment_export_filename))


class Real_Time_Eye_Movement_Detector(Observable, _Eye_Movement_Detector_Base):
    """Eye movement classification detector based on segmented linear regression.
    """

    MENU_LABEL_TEXT = "Eye Movement Detector"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._buffered_detection = Eye_Movement_Buffered_Detection()
        self._recent_segments = []

    def recent_events(self, events):

        if not self.__profile_is_enabled:
            self.__profile.enable()
        self.__profile_is_enabled = True

        gaze_data = events["gaze"]
        capture = Immutable_Capture(self.g_pool.capture)

        self._buffered_detection.extend_gaze_data(gaze_data=gaze_data, capture=capture)

        frame_timestamp = events['frame'].timestamp
        self._recent_segments = self._buffered_detection.segments_at_timestamp(frame_timestamp)

        public_segments = [ segment.to_public_dict() for segment in self._recent_segments ]
        events[EYE_MOVEMENT_EVENT_KEY] = public_segments

    def gl_display(self):
        frame_size = self.g_pool.capture.frame_size
        for segment in self._recent_segments:
            segment.draw_in_gl_context(frame_size, self.glfont)

    def init_ui(self):
        self.add_menu()
        self.menu.label = type(self).MENU_LABEL_TEXT

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(ui.Info_Text(help_str))

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())

    def deinit_ui(self):
        self.remove_menu()
        self.glfont = None

    def get_init_dict(self):
        return {
        }

