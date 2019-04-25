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
import typing
import collections
from .eye_movement_detector_base import Eye_Movement_Detector_Base
from eye_movement.utils import Gaze_Data, EYE_MOVEMENT_EVENT_KEY, logger
from eye_movement.model.immutable_capture import Immutable_Capture
from eye_movement.model.segment import Classified_Segment
from eye_movement.worker.offline_detection_task import Offline_Detection_Task
from eye_movement.model.storage import Classified_Segment_Storage
from eye_movement.controller.eye_movement_csv_exporter import Eye_Movement_CSV_Exporter
from eye_movement.controller.eye_movement_seek_controller import Eye_Movement_Seek_Controller
from eye_movement.ui.menu_content import Menu_Content
from eye_movement.ui.navigation_buttons import Prev_Segment_Button, Next_Segment_Button
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
        self.eye_movement_detection_yields = collections.deque()

        self.task_manager = PluginTaskManager(self)
        self.eye_movement_task = None

        self.notify_all(
            {"subject": Notification_Subject.SHOULD_RECALCULATE, "delay": 0.5}
        self.storage = Classified_Segment_Storage(
            plugin=self,
            rec_dir=g_pool.rec_dir,
        )
        self.seek_controller = Eye_Movement_Seek_Controller(
            plugin=self,
            storage=self.storage,
            seek_to_timestamp=self.seek_to_timestamp,
        )

    def init_ui(self):
        )
        self.menu_content = Menu_Content(
            plugin=self,
            label_text=self.MENU_LABEL_TEXT,
            show_segmentation=show_segmentation,
        )
        self.prev_segment_button = Prev_Segment_Button(
            on_click=self.seek_controller.jump_to_prev_segment
        )
        self.next_segment_button = Next_Segment_Button(
            on_click=self.seek_controller.jump_to_next_segment
        )

        )
        )
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



    def init_ui(self):
        self.add_menu()
        self.menu_content.add_to_menu(self.menu)
        self.prev_segment_button.add_to_quickbar(self.g_pool.quickbar)
        self.next_segment_button.add_to_quickbar(self.g_pool.quickbar)

        if len(self.storage):
            status = "Loaded from cache"
            self.menu_content.update_status(status)
        else:
            self.trigger_recalculate()

    def deinit_ui(self):
        self.remove_menu()
        self.prev_segment_button.remove_from_quickbar(self.g_pool.quickbar)
        self.next_segment_button.remove_from_quickbar(self.g_pool.quickbar)

    def get_init_dict(self):
        return {
            "show_segmentation": self.menu_content.show_segmentation,
        }


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



    def export_eye_movement(self, export_range, export_dir):

        segments_in_section = self.storage.segments_in_range(export_range)

        if segments_in_section:
            csv_exporter = Eye_Movement_CSV_Exporter()
            csv_exporter.csv_export(segments_in_section, export_dir=export_dir)
        else:
            logger.warning("No fixations in this recording nothing to export")
