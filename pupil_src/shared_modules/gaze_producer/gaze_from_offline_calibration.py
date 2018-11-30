"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from gaze_producer.controller.reference_location_controllers import (
    ReferenceDetectionController,
    ReferenceEditController,
)
from gaze_producer.model.reference_location_storage import ReferenceLocationStorage
from gaze_producer.ui.reference_location_renderer import ReferenceLocationRenderer
from gaze_producer.ui.reference_location_menu import ReferenceLocationMenu
from gaze_producer.ui.timeline import Timeline
from observable import Observable
from plugin import Analysis_Plugin_Base
from plugin_timeline import PluginTimeline
from tasklib.manager import PluginTaskManager


class GazeFromOfflineCalibration(Analysis_Plugin_Base, Observable):
    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.inject_plugin_dependencies()

        self._task_manager = PluginTaskManager(plugin=self)

        self._reference_location_storage = ReferenceLocationStorage(
            self.g_pool.rec_dir, plugin=self
        )

        self._reference_detection_controller = ReferenceDetectionController(
            self._task_manager, self._reference_location_storage
        )
        self._reference_edit_controller = ReferenceEditController(
            self._reference_location_storage,
            plugin=self,
            all_timestamps=self.g_pool.timestamps,
            get_current_frame_index=self.g_pool.capture.get_frame_index,
            seek_to_frame=self._seek_to_frame,
        )

        self._reference_location_renderer = ReferenceLocationRenderer(
            self._reference_location_storage,
            plugin=self,
            frame_size=self.g_pool.capture.frame_size,
            get_current_frame_index=self.g_pool.capture.get_frame_index,
        )
        self._reference_location_menu = ReferenceLocationMenu(
            self._reference_detection_controller,
            self._reference_location_storage,
            self._reference_edit_controller,
        )

        plugin_timeline = PluginTimeline(
            title="Offline Calibration",
            plugin=self,
            user_timelines=self.g_pool.user_timelines,
            time_start=self.g_pool.timestamps[0],
            time_end=self.g_pool.timestamps[-1],
        )
        self._timeline = Timeline(
            plugin_timeline,
            self._reference_detection_controller,
            self._reference_location_storage,
            plugin=self,
        )

    def inject_plugin_dependencies(self):
        from gaze_producer.worker.detect_circle_markers import CircleMarkerDetectionTask

        CircleMarkerDetectionTask.zmq_ctx = self.g_pool.zmq_ctx
        CircleMarkerDetectionTask.capture_source_path = self.g_pool.capture.source_path
        CircleMarkerDetectionTask.notify_all = self.notify_all

    def init_ui(self):
        super().init_ui()
        self.add_menu()
        self.menu.label = "Gaze From Offline Calibration"
        self._reference_location_menu.render()
        self.menu.append(self._reference_location_menu.menu)

    def _seek_to_frame(self, frame_index):
        self.notify_all({"subject": "seek_control.should_seek", "index": frame_index})
