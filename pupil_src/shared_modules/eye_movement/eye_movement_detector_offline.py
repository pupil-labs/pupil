"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import random
from .eye_movement_detector_base import Eye_Movement_Detector_Base
import eye_movement.utils as utils
import eye_movement.model as model
import eye_movement.controller as controller
import eye_movement.ui as ui
from observable import Observable
from data_changed import Listener, Announcer


logger = logging.getLogger(__name__)


class Notification_Subject:
    SHOULD_RECALCULATE = "segmentation_detector.should_recalculate"
    MIN_DATA_CONFIDENCE_CHANGED = "min_data_confidence_changed"


EYE_MOVEMENT_ANNOUNCER_TOPIC = "eye_movement"


class Offline_Eye_Movement_Detector(Observable, Eye_Movement_Detector_Base):
    """
    Eye movement classification detector based on segmented linear regression.

    Event identification is based on segmentation that simultaneously denoises the signal and determines event
    boundaries. The full gaze position time-series is segmented into an approximately optimal piecewise linear
    function in O(n) time. Gaze feature parameters for classification into fixations, saccades, smooth pursuits and post-saccadic oscillations
    are derived from human labeling in a data-driven manner.

    More details about this approach can be found here:
    https://www.nature.com/articles/s41598-017-17983-x

    The open source implementation can be found here:
    https://gitlab.com/nslr/nslr-hmm
    """

    MENU_LABEL_TEXT = "Eye Movement Detector"

    def __init__(self, g_pool, show_segmentation=True):
        super().__init__(g_pool)
        self.storage = model.Classified_Segment_Storage(
            plugin=self, rec_dir=g_pool.rec_dir
        )
        self.seek_controller = controller.Eye_Movement_Seek_Controller(
            plugin=self, storage=self.storage, seek_to_timestamp=self.seek_to_timestamp
        )
        self.offline_controller = controller.Eye_Movement_Offline_Controller(
            plugin=self,
            storage=self.storage,
            on_started=self.on_task_started,
            on_status=self.on_task_status,
            on_progress=self.on_task_progress,
            on_exception=self.on_task_exception,
            on_completed=self.on_task_completed,
        )
        self.menu_content = ui.Menu_Content(
            plugin=self,
            label_text=self.MENU_LABEL_TEXT,
            show_segmentation=show_segmentation,
        )
        self.prev_segment_button = ui.Prev_Segment_Button(
            on_click=self.seek_controller.jump_to_prev_segment
        )
        self.next_segment_button = ui.Next_Segment_Button(
            on_click=self.seek_controller.jump_to_next_segment
        )
        self._gaze_changed_listener = Listener(
            plugin=self, topic="gaze_positions", rec_dir=g_pool.rec_dir
        )
        self._gaze_changed_listener.add_observer(
            method_name="on_data_changed", observer=self.offline_controller.classify
        )
        self._eye_movement_changed_announcer = Announcer(
            plugin=self, topic=EYE_MOVEMENT_ANNOUNCER_TOPIC, rec_dir=g_pool.rec_dir
        )

    #

    def trigger_recalculate(self):
        self.notify_all(
            {"subject": Notification_Subject.SHOULD_RECALCULATE, "delay": 0.5}
        )

    def seek_to_timestamp(self, timestamp):
        self.notify_all({"subject": "seek_control.should_seek", "timestamp": timestamp})

    def on_task_started(self):
        self.menu_content.update_error_text("")

    def on_task_progress(self, progress: float):
        self.menu_content.update_progress(progress)

    def on_task_status(self, status: str):
        self.menu_content.update_status(status)

    def on_task_exception(self, exception: Exception):
        error_message = f"{exception}"
        logger.error(error_message)
        self.menu_content.update_error_text(error_message)

    def on_task_completed(self):
        self._eye_movement_changed_announcer.announce_new()

    #

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
        return {"show_segmentation": self.menu_content.show_segmentation}

    def on_notify(self, notification):
        if notification["subject"] == "gaze_positions_changed":
            # TODO: Remove when gaze_positions will be announced with `data_changed.Announcer`
            note = notification.copy()
            note["subject"] = "data_changed.{}.announce_token".format(
                self._gaze_changed_listener._topic
            )
            note["token"] = notification.get(
                "token", "{:0>8x}".format(random.getrandbits(32))
            )
            self._gaze_changed_listener._on_notify(note)
        elif notification["subject"] in (
            Notification_Subject.SHOULD_RECALCULATE,
            Notification_Subject.MIN_DATA_CONFIDENCE_CHANGED,
        ):
            self.offline_controller.classify()
        elif notification["subject"] == "should_export":
            self.export_eye_movement(
                notification["ts_window"], notification["export_dir"]
            )

    def recent_events(self, events):

        frame = events.get("frame")
        if not frame:
            return

        visible_segments = self.storage.segments_in_frame(frame)
        self.seek_controller.update_visible_segments(visible_segments)

        self.menu_content.update_detail_text(
            current_index=self.seek_controller.current_segment_index,
            total_segment_count=self.seek_controller.total_segment_count,
            current_segment=self.seek_controller.current_segment,
            prev_segment=self.seek_controller.prev_segment,
            next_segment=self.seek_controller.next_segment,
        )

        if self.menu_content.show_segmentation:
            segment_renderer = ui.Segment_Overlay_Image_Renderer(
                canvas_size=(frame.width, frame.height), image=frame.img
            )
            for segment in visible_segments:
                segment_renderer.draw(segment)

        events[utils.EYE_MOVEMENT_EVENT_KEY] = visible_segments

    def export_eye_movement(self, export_window, export_dir):

        segments_in_section = self.storage.segments_in_timestamp_window(export_window)

        if segments_in_section:
            by_segment_csv_exporter = controller.Eye_Movement_By_Segment_CSV_Exporter()
            by_segment_csv_exporter.csv_export(
                segments_in_section, export_dir=export_dir
            )

            export_window_start, export_window_stop = export_window
            ts_segment_class_pairs = (
                (gaze["timestamp"], seg.segment_class)
                for seg in segments_in_section
                for gaze in seg.segment_data
                if export_window_start <= gaze["timestamp"] <= export_window_stop
            )
            by_gaze_csv_exporter = controller.Eye_Movement_By_Gaze_CSV_Exporter()
            by_gaze_csv_exporter.csv_export(
                ts_segment_class_pairs, export_dir=export_dir
            )
        else:
            logger.warning(
                "The selected export range does not include eye movement detections"
            )
