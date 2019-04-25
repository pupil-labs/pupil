"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import csv
import typing
import collections
from .eye_movement_detector_base import Eye_Movement_Detector_Base
from eye_movement.utils import Gaze_Data, EYE_MOVEMENT_EVENT_KEY, logger
from eye_movement.model.immutable_capture import Immutable_Capture
from eye_movement.model.segment import Classified_Segment
from eye_movement.worker.offline_detection_task import Offline_Detection_Task
from eye_movement.model.storage import Classified_Segment_Storage
from observable import Observable
from tasklib.manager import PluginTaskManager
import player_methods as pm
from pyglui import ui


class Notification_Subject:
    SHOULD_RECALCULATE = "segmentation_detector.should_recalculate"
    SEGMENTATION_CHANGED = "segmentation_changed"


class Offline_Eye_Movement_Detector(Observable, Eye_Movement_Detector_Base):
    """
    Eye movement classification detector based on segmented linear regression.
    """

    MENU_LABEL_TEXT = "Eye Movement Detector"

    def __init__(self, g_pool, show_segmentation=True):
        super().__init__(g_pool)
        self.show_segmentation = show_segmentation
        self.current_segment_index = None
        self.eye_movement_detection_yields = collections.deque()
        self.status = ""

        self.task_manager = PluginTaskManager(self)
        self.eye_movement_task = None

        self.current_segment_details = None
        self.next_segment_button = None
        self.prev_segment_button = None

        self.notify_all(
            {"subject": Notification_Subject.SHOULD_RECALCULATE, "delay": 0.5}
        self.storage = Classified_Segment_Storage(
            plugin=self,
            rec_dir=g_pool.rec_dir,
        )
        )

    def init_ui(self):
        self.add_menu()
        self.menu.label = type(self).MENU_LABEL_TEXT

        def jump_next_segment(_):
            if len(self.g_pool.eye_movements) < 1:
                logger.warning("No eye movement segments availabe")
                return

            # Set current segment index to next one, or to 0 if not available
            self.current_segment_index = (
                self.current_segment_index if self.current_segment_index else 0
            )
            self.current_segment_index = (self.current_segment_index + 1) % len(
                self.g_pool.eye_movements
            )

            next_segment_ts = self.g_pool.eye_movements[
                self.current_segment_index
            ].start_frame_timestamp

            self.notify_all(
                {
                    "subject": "seek_control.should_seek",
                    "timestamp": next_segment_ts,
                }
            )

        def jump_prev_segment(_):
            if len(self.g_pool.eye_movements) < 1:
                logger.warning("No segmentation availabe")
                return

            # Set current segment index to previous one, or to 0 if not available
            self.current_segment_index = (
                self.current_segment_index if self.current_segment_index else 0
            )
            self.current_segment_index = (self.current_segment_index - 1) % len(
                self.g_pool.eye_movements
            )

            prev_segment_ts = self.g_pool.eye_movements[
                self.current_segment_index
            ].start_frame_timestamp

            self.notify_all(
                {
                    "subject": "seek_control.should_seek",
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

        self.eye_movement_task = Offline_Detection_Task(args=(capture, gaze_data))
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

    def on_task_exception(self, exception: Exception):
        raise exception

    def on_task_completed(self, _: None):
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
            self.g_pool.eye_movements[index - 1] if index > 0 else None
        )
        next_segment = (
            self.g_pool.eye_movements[self.current_segment_index + 1]
            if self.current_segment_index < len(self.g_pool.eye_movements) - 1
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

    def _find_focused_segment(self, visible_segments):
        current_segment = None
        visible_segments = visible_segments if visible_segments else []
        current_segment_index = self.current_segment_index

        if current_segment_index:
            current_segment_index = current_segment_index % len(
                self.g_pool.eye_movements
            )
            current_segment = self.g_pool.eye_movements[
                current_segment_index
            ]

        if not visible_segments:
            return current_segment_index, current_segment

        if (current_segment not in visible_segments) and len(visible_segments) > 0:
            current_segment = visible_segments[0]
            current_segment_index = self.g_pool.eye_movements.data.index(
                current_segment
            )

        return current_segment_index, current_segment

    def recent_events(self, events):

        frame = events.get("frame")
        if not frame:
            return

        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
        visible_segments: typing.Iterable[
            Classified_Segment
        ] = self.g_pool.eye_movements.by_ts_window(frame_window)
        events[EYE_MOVEMENT_EVENT_KEY] = visible_segments

        self.current_segment_index, current_segment = self._find_focused_segment(visible_segments)

        self._ui_draw_visible_segments(frame, visible_segments)
        self._ui_update_segment_detail_text(
            self.current_segment_index,
            len(self.g_pool.eye_movements),
            current_segment,
        )

    def correlate_and_publish(self):
        self.g_pool.eye_movements = pm.Affiliator(
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
        segments_in_section = self.g_pool.eye_movements.by_ts_window(
            export_window
        )

        segment_export_filename = "eye_movement.csv"
        segment_export_full_path = os.path.join(export_dir, segment_export_filename)

        with open(
            segment_export_full_path, "w", encoding="utf-8", newline=""
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(type(self).csv_header())
            for segment in segments_in_section:
                csv_writer.writerow(type(self).csv_row(segment))
            logger.info("Created '{}' file.".format(segment_export_filename))
