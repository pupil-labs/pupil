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

import glfw
import numpy as np
from gl_utils import GLFWErrorReporting

GLFWErrorReporting.set_default()

import tasklib
from gaze_producer import model, worker
from observable import Observable

logger = logging.getLogger(__name__)


class ReferenceDetectionController(Observable):
    def __init__(self, task_manager, reference_location_storage):
        self._task_manager = task_manager
        self._reference_location_storage = reference_location_storage
        self._detection_task = None

    def start_detection(self):
        def on_detection_yields(detection):
            self._reference_location_storage.add(detection)

        def on_detection_completed(_):
            self._reference_location_storage.save_to_disk()

        self._detection_task = worker.detect_circle_markers.CircleMarkerDetectionTask()
        self._detection_task.add_observer("on_exception", tasklib.raise_exception)
        self._detection_task.add_observer(
            "on_started", lambda: self.on_detection_started(self._detection_task)
        )
        self._detection_task.add_observer("on_yield", on_detection_yields)
        self._detection_task.add_observer("on_completed", on_detection_completed)
        self._task_manager.add_task(
            self._detection_task, identifier="reference_detection"
        )
        return self._detection_task

    def on_detection_started(self, detection_task):
        """By observing this, other modules can add their own observers to the task"""
        pass

    def cancel_detection(self):
        if not self._detection_task:
            raise ValueError("No detection task running!")
        self._detection_task.cancel_gracefully()

    @property
    def is_running_detection(self):
        return self._detection_task is not None and self._detection_task.running

    @property
    def detection_progress(self):
        return self._detection_task.progress if self._detection_task else 0.0


class ReferenceEditController:
    def __init__(
        self,
        reference_location_storage,
        all_timestamps,
        get_current_frame_index,
        seek_to_frame,
        plugin,
    ):
        self.edit_mode_active = False

        self._reference_location_storage = reference_location_storage

        self._all_timestamps = all_timestamps
        self._get_current_frame_index = get_current_frame_index
        self._seek_to_frame = seek_to_frame

        plugin.add_observer("on_click", self._on_click)

    def jump_to_next_ref(self):
        try:
            current_index = self._get_current_frame_index()
            next_ref = self._reference_location_storage.get_next(current_index)
        except ValueError:
            logger.warning("Could not jump, no next reference location found!")
        else:
            self._seek_to_frame(next_ref.frame_index)

    def jump_to_prev_ref(self):
        try:
            current_index = self._get_current_frame_index()
            prev_ref = self._reference_location_storage.get_previous(current_index)
        except ValueError:
            logger.warning("Could not jump, no previous reference location found!")
        else:
            self._seek_to_frame(prev_ref.frame_index)

    def _on_click(self, pos, button, action):
        if action == glfw.PRESS:
            self._add_or_delete_ref_on_click(pos)

    def _add_or_delete_ref_on_click(self, pos):
        if not self.edit_mode_active:
            return
        current_reference = self._get_reference_for_current_frame()
        if self._clicked_on_reference(current_reference, pos):
            self._reference_location_storage.delete(current_reference)
        else:
            self._add_reference(pos)
        self._reference_location_storage.save_to_disk()

    def _get_reference_for_current_frame(self):
        current_index = self._get_current_frame_index()
        return self._reference_location_storage.get_or_none(current_index)

    @staticmethod
    def _clicked_on_reference(reference_location, click_pos):
        if reference_location is None:
            return False
        pos_x = click_pos[0]
        pos_y = click_pos[1]
        click_distance = np.sqrt(
            (pos_x - reference_location.screen_x) ** 2
            + (pos_y - reference_location.screen_y) ** 2
        )
        return click_distance < 15

    def _add_reference(self, pos):
        screen_pos = tuple(pos)
        frame_index = self._get_current_frame_index()
        timestamp = self._all_timestamps[frame_index]
        reference_location = model.ReferenceLocation(screen_pos, frame_index, timestamp)
        self._reference_location_storage.add(reference_location)
