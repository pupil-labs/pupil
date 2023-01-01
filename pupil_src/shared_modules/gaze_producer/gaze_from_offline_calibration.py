"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import data_changed
from gaze_producer import controller, model
from gaze_producer import ui as plugin_ui
from gaze_producer.gaze_producer_base import GazeProducerBase
from plugin_timeline import PluginTimeline
from pupil_recording import PupilRecording, RecordingInfo
from tasklib.manager import UniqueTaskManager


# IMPORTANT: GazeProducerBase needs to be THE LAST in the list of bases, otherwise
# uniqueness by base class does not work
class GazeFromOfflineCalibration(GazeProducerBase):
    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"

    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE
            ):
                # Disable post-hoc gaze calibration in Player if Pupil Invisible recording
                return False
        return super().is_available_within_context(g_pool)

    @classmethod
    def plugin_menu_label(cls) -> str:
        return "Post-Hoc Gaze Calibration"

    @classmethod
    def gaze_data_source_selection_order(cls) -> float:
        return 2.0

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.inject_plugin_dependencies()

        self._task_manager = UniqueTaskManager(plugin=self)

        self._recording_uuid = PupilRecording(g_pool.rec_dir).meta_info.recording_uuid

        self._setup_storages()
        self._setup_controllers()
        self._setup_ui()
        self._setup_timelines()

        self._pupil_changed_listener = data_changed.Listener(
            "pupil_positions", g_pool.rec_dir, plugin=self
        )
        self._pupil_changed_listener.add_observer(
            "on_data_changed",
            self._calculate_all_controller.calculate_all_if_references_available,
        )

    def _setup_storages(self):
        self._reference_location_storage = model.ReferenceLocationStorage(
            self.g_pool.rec_dir
        )
        self._calibration_storage = model.CalibrationStorage(
            rec_dir=self.g_pool.rec_dir,
            get_recording_index_range=self._recording_index_range,
            recording_uuid=self._recording_uuid,
        )
        self._gaze_mapper_storage = model.GazeMapperStorage(
            self._calibration_storage,
            rec_dir=self.g_pool.rec_dir,
            get_recording_index_range=self._recording_index_range,
        )

    def cleanup(self):
        super().cleanup()
        self._reference_location_storage.save_to_disk()
        self._calibration_storage.save_to_disk()
        self._gaze_mapper_storage.save_to_disk()

    def _setup_controllers(self):
        self._reference_detection_controller = controller.ReferenceDetectionController(
            self._task_manager, self._reference_location_storage
        )
        self._reference_edit_controller = controller.ReferenceEditController(
            self._reference_location_storage,
            plugin=self,
            all_timestamps=self.g_pool.timestamps,
            get_current_frame_index=self.g_pool.capture.get_frame_index,
            seek_to_frame=self._seek_to_frame,
        )
        self._calibration_controller = controller.CalibrationController(
            self._calibration_storage,
            self._reference_location_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            recording_uuid=self._recording_uuid,
        )
        self._gaze_mapper_controller = controller.GazeMapperController(
            self._gaze_mapper_storage,
            self._calibration_storage,
            self._reference_location_storage,
            task_manager=self._task_manager,
            get_current_trim_mark_range=self._current_trim_mark_range,
            publish_gaze_bisector=self._publish_gaze,
        )
        self._calculate_all_controller = controller.CalculateAllController(
            self._reference_detection_controller,
            self._reference_location_storage,
            self._calibration_controller,
            self._calibration_storage,
            self._gaze_mapper_controller,
            self._gaze_mapper_storage,
        )

    def _setup_ui(self):
        self._reference_location_renderer = plugin_ui.ReferenceLocationRenderer(
            self._reference_location_storage,
            plugin=self,
            frame_size=self.g_pool.capture.frame_size,
            get_current_frame_index=self.g_pool.capture.get_frame_index,
        )
        self._on_top_menu = plugin_ui.OnTopMenu(
            self._calculate_all_controller, self._reference_location_storage
        )
        self._reference_location_menu = plugin_ui.ReferenceLocationMenu(
            self._reference_detection_controller,
            self._reference_location_storage,
            self._reference_edit_controller,
        )
        self._calibration_menu = plugin_ui.CalibrationMenu(
            self._calibration_storage,
            self._calibration_controller,
            index_range_as_str=self._index_range_as_str,
        )
        self._gaze_mapper_menu = plugin_ui.GazeMapperMenu(
            self._gaze_mapper_controller,
            self._gaze_mapper_storage,
            self._calibration_storage,
            index_range_as_str=self._index_range_as_str,
        )

    def _setup_timelines(self):
        self._reference_location_timeline = plugin_ui.ReferenceLocationTimeline(
            self._reference_detection_controller, self._reference_location_storage
        )
        self._gaze_mapper_timeline = plugin_ui.GazeMapperTimeline(
            self._gaze_mapper_storage,
            self._gaze_mapper_controller,
            self._calibration_storage,
            self._calibration_controller,
        )
        plugin_timeline = PluginTimeline(
            plugin=self,
            title="Offline Calibration",
            timeline_ui_parent=self.g_pool.user_timelines,
            all_timestamps=self.g_pool.timestamps,
        )
        self._timeline = plugin_ui.OfflineCalibrationTimeline(
            plugin_timeline,
            self._reference_location_timeline,
            self._gaze_mapper_timeline,
            plugin=self,
        )

    def inject_plugin_dependencies(self):
        from gaze_producer.worker.detect_circle_markers import CircleMarkerDetectionTask

        CircleMarkerDetectionTask.zmq_ctx = self.g_pool.zmq_ctx
        CircleMarkerDetectionTask.capture_source_path = self.g_pool.capture.source_path
        CircleMarkerDetectionTask.notify_all = self.notify_all

        from gaze_producer.worker import create_calibration

        create_calibration.g_pool = self.g_pool

        from gaze_producer.worker import map_gaze

        map_gaze.g_pool = self.g_pool

        from gaze_producer.worker import validate_gaze

        validate_gaze.g_pool = self.g_pool

    def init_ui(self):
        super().init_ui()
        self._on_top_menu.render(self.menu)
        self._reference_location_menu.render()
        self.menu.append(self._reference_location_menu.menu)
        self._calibration_menu.render()
        self.menu.append(self._calibration_menu.menu)
        self._gaze_mapper_menu.render()
        self.menu.append(self._gaze_mapper_menu.menu)

    def _publish_gaze(self, gaze_bisector):
        self.g_pool.gaze_positions = gaze_bisector
        self._gaze_changed_announcer.announce_new(delay=1)

    def _seek_to_frame(self, frame_index):
        self.notify_all({"subject": "seek_control.should_seek", "index": frame_index})

    def _recording_index_range(self):
        left_index = 0
        right_index = len(self.g_pool.timestamps) - 1
        return left_index, right_index

    def _current_trim_mark_range(self):
        right_idx = self.g_pool.seek_control.trim_right
        left_idx = self.g_pool.seek_control.trim_left
        return left_idx, right_idx

    def _index_range_as_str(self, index_range):
        from_index, to_index = index_range
        return (
            f"{self._index_time_as_str(from_index)} - "
            f"{self._index_time_as_str(to_index)}"
        )

    def _index_time_as_str(self, index):
        ts = self.g_pool.timestamps[index]
        min_ts = self.g_pool.timestamps[0]
        time = ts - min_ts
        minutes = abs(time // 60)  # abs because it's sometimes -0
        seconds = round(time % 60)
        return f"{minutes:02.0f}:{seconds:02.0f}"
