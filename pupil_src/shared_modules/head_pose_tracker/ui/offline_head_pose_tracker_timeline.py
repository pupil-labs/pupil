"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from plugin_timeline import BarsElementTs, RangeElementFrameIdx, Row


class OfflineHeadPoseTrackerTimeline:
    def __init__(
        self, plugin_timeline, detection_timeline, localization_timeline, plugin
    ):
        self._plugin_timeline = plugin_timeline
        self._detection_timeline = detection_timeline
        self._localization_timeline = localization_timeline
        self._plugin = plugin

        plugin.add_observer("init_ui", self._on_init_ui)

        detection_timeline.render_parent_timeline = self.render
        localization_timeline.render_parent_timeline = self.render

    def _on_init_ui(self):
        self.render()

    def render(self):
        self._plugin_timeline.clear_rows()
        self._plugin_timeline.add_row(self._detection_timeline.row)
        self._plugin_timeline.add_row(self._localization_timeline.row)
        self._plugin_timeline.refresh()


class DetectionTimeline:
    timeline_label = "Marker detection"

    def __init__(
        self, detection_controller, general_settings, detection_storage, all_timestamps
    ):
        self.render_parent_timeline = None

        self._detection_controller = detection_controller
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._all_timestamps = all_timestamps

        detection_storage.add_observer(
            "load_pldata_from_disk", self._on_storage_changed
        )
        detection_controller.add_observer(
            "on_detection_started", self._on_detection_started
        )
        detection_controller.add_observer(
            "on_detection_yield", self._on_detection_yield
        )
        detection_controller.add_observer(
            "on_detection_ended", self._on_detection_ended
        )
        self.row = None
        self.update_row()

    def update_row(self):
        elements = [self._create_detection_bars()]
        if self._detection_controller.is_running_task:
            elements.append(self._create_progress_indication())

        self.row = Row(label=self.timeline_label, elements=elements)

    def _create_detection_bars(self):
        frame_indices = [
            frame_index
            for frame_index in self._detection_storage.frame_index_to_num_markers
            if self._detection_storage.frame_index_to_num_markers[frame_index] > 0
        ]
        bar_positions = self._all_timestamps[frame_indices]
        return BarsElementTs(
            bar_positions, color_rgba=(0.7, 0.3, 0.2, 0.33), width=1, height=12
        )

    def _create_progress_indication(self):
        progress = self._detection_controller.progress
        if progress > 0:
            return RangeElementFrameIdx(
                from_idx=self._frame_start,
                to_idx=int(self._frame_start + self._frame_count * progress) - 1,
                color_rgba=(1.0, 1.0, 1.0, 0.8),
                height=4,
            )
        else:
            return RangeElementFrameIdx()

    def _on_detection_started(self):
        (
            self._frame_start,
            frame_end,
        ) = self._general_settings.detection_frame_index_range
        self._frame_count = frame_end - self._frame_start + 1

    def _on_storage_changed(self):
        self.update_row()
        self.render_parent_timeline()

    def _on_detection_yield(self):
        self.update_row()
        self.render_parent_timeline()

    def _on_detection_ended(self):
        self.update_row()
        self.render_parent_timeline()


class LocalizationTimeline:
    timeline_label = "Camera localization"

    def __init__(self, localization_controller, general_settings, localization_storage):
        self.render_parent_timeline = None

        self._localization_controller = localization_controller
        self._general_settings = general_settings
        self._localization_storage = localization_storage

        localization_storage.add_observer(
            "load_pldata_from_disk", self._on_storage_changed
        )
        localization_controller.add_observer("reset", self._on_localization_reset)
        localization_controller.add_observer(
            "on_localization_started", self._on_localization_started
        )
        localization_controller.add_observer(
            "on_localization_yield", self._on_localization_yield
        )
        localization_controller.add_observer(
            "on_localization_ended", self._on_localization_ended
        )
        self.row = None
        self.update_row()

    def update_row(self):
        elements = [self._create_localization_bars()]
        if self._localization_controller.is_running_task:
            elements.append(self._create_progress_indication())

        self.row = Row(label=self.timeline_label, elements=elements)

    def _create_localization_bars(self):
        bar_positions = self._localization_storage.pose_bisector.timestamps
        return BarsElementTs(
            bar_positions, color_rgba=(0.0, 0.5, 1.0, 0.8), width=1, height=12
        )

    def _create_progress_indication(self):
        progress = self._localization_controller.progress
        if progress > 0:
            return RangeElementFrameIdx(
                from_idx=self._frame_start,
                to_idx=int(self._frame_start + self._frame_count * progress) - 1,
                color_rgba=(1.0, 1.0, 1.0, 0.8),
                height=4,
            )
        else:
            return RangeElementFrameIdx()

    def _on_localization_reset(self):
        self.update_row()
        self.render_parent_timeline()

    def _on_localization_started(self):
        (
            self._frame_start,
            frame_end,
        ) = self._general_settings.localization_frame_index_range
        self._frame_count = frame_end - self._frame_start + 1

    def _on_storage_changed(self):
        self.update_row()
        self.render_parent_timeline()

    def _on_localization_yield(self):
        self.update_row()
        self.render_parent_timeline()

    def _on_localization_ended(self):
        self.update_row()
        self.render_parent_timeline()
