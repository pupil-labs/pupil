"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

import numpy as np
import glfw

import tasklib
from gaze_producer.worker.detect_circle_markers import CircleMarkerDetectionTask
from observable import Observable

logger = logging.getLogger(__name__)


class ReferenceDetectionController(Observable):
    def __init__(self, task_manager, reference_location_storage):
        self._task_manager = task_manager
        self._reference_location_storage = reference_location_storage
        self.task = None

    def start_detection(self):
        self.task = CircleMarkerDetectionTask()
        self.task.add_observer("on_exception", tasklib.raise_exception)
        self.task.add_observer("on_yield", self._on_detection_yields)
        self._task_manager.add_task(self.task)
        return self.task

    def _on_detection_yields(self, detection):
        self._reference_location_storage.add(
            detection.screen_pos, detection.frame_index, detection.timestamp
        )

    def cancel_detection(self):
        if not self.task:
            raise ValueError("No detection task running!")
        self.task.cancel_gracefully()

    @property
    def is_running_detection(self):
        return self.task is not None and self.task.running


class ReferenceEditController:
    def __init__(
        self,
        reference_location_storage,
        plugin,
        all_timestamps,
        get_current_frame_index,
        seek_to_frame,
    ):
        self.edit_mode_active = True

        self._reference_location_storage = reference_location_storage

        self._all_timestamps = all_timestamps
        self._get_current_frame_index = get_current_frame_index
        self._seek_to_frame = seek_to_frame

        plugin.add_observer("on_click", self.on_click)

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

    def on_click(self, pos, button, action):
        if not self.edit_mode_active:
            return
        if action == glfw.GLFW_PRESS:
            current_reference = self._get_current_reference()
            if self._clicked_on_reference(current_reference, pos):
                self._reference_location_storage.delete(current_reference)
            else:
                self._add_reference(pos)

    def _get_current_reference(self):
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
        self._reference_location_storage.add(screen_pos, frame_index, timestamp)
