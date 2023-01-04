"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from plugin_timeline import BarsElementTs, RangeElementFramePerc, Row


class ReferenceLocationTimeline:
    def __init__(self, reference_detection_controller, reference_location_storage):
        self.render_parent_timeline = None

        self._reference_detection_controller = reference_detection_controller
        self._reference_location_storage = reference_location_storage

        self._reference_detection_controller.add_observer(
            "on_detection_started", self._on_start_reference_detection
        )
        self._reference_location_storage.add_observer(
            "add", self._on_reference_storage_changed
        )
        self._reference_location_storage.add_observer(
            "delete", self._on_reference_storage_changed
        )
        self._reference_location_storage.add_observer(
            "delete_all", self._on_reference_storage_changed
        )

    def create_row(self):
        elements = []
        if self._reference_detection_controller.is_running_detection:
            elements.append(self._create_progress_indication())
        elements.append(self._create_reference_location_bars())
        return Row(label="References", elements=elements)

    def _create_progress_indication(self):
        progress = self._reference_detection_controller.detection_progress
        return RangeElementFramePerc(
            from_perc=0, to_perc=progress, color_rgba=(1.0, 0.5, 0.5, 0.5)
        )

    def _create_reference_location_bars(self):
        bar_positions = [ref.timestamp for ref in self._reference_location_storage]
        return BarsElementTs(bar_positions, color_rgba=(1.0, 1.0, 1.0, 0.5))

    def _on_start_reference_detection(self, detection_task):
        detection_task.add_observer("update", self._on_reference_detection_update)
        detection_task.add_observer("on_ended", self._on_reference_detection_ended)

    def _on_reference_detection_update(self):
        self.render_parent_timeline()

    def _on_reference_detection_ended(self):
        self.render_parent_timeline()

    def _on_reference_storage_changed(self, *args, **kwargs):
        self.render_parent_timeline()
