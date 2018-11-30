"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from plugin_timeline import Line, BarsElementTs, RangeElementPerc


class Timeline:
    def __init__(
        self,
        plugin_timeline,
        reference_detection_controller,
        reference_location_storage,
        plugin,
    ):
        self._plugin_timeline = plugin_timeline
        self._reference_detection_controller = reference_detection_controller
        self._reference_location_storage = reference_location_storage

        plugin.add_observer("init_ui", self._on_init_ui)
        self._reference_detection_controller.add_observer(
            "start_detection", self._on_start_reference_detection
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

    def _on_init_ui(self):
        self.render()

    def _on_start_reference_detection(self):
        self._reference_detection_controller.task.add_observer(
            "update", self._on_reference_detection_update
        )
        self._reference_detection_controller.task.add_observer(
            "on_ended", self._on_reference_detection_ended
        )

    def _on_reference_detection_update(self):
        self.render()

    def _on_reference_detection_ended(self):
        self.render()

    def _on_reference_storage_changed(self, *args, **kwargs):
        self.render()

    def render(self):
        self._plugin_timeline.clear_lines()
        self._plugin_timeline.add_line(self._render_references_timeline())
        self._plugin_timeline.refresh()

    def _render_references_timeline(self):
        elements = []
        if self._reference_detection_controller.is_running_detection:
            elements.append(self._create_progress_indication())
        elements.append(self._create_reference_location_bars())
        return Line(label="References", elements=elements)

    def _create_progress_indication(self):
        progress = self._reference_detection_controller.task.progress
        return RangeElementPerc(
            from_perc=0, to_perc=progress, color_rgba=(1.0, 0.5, 0.5, 0.5)
        )

    def _create_reference_location_bars(self):
        bar_positions = [ref.timestamp for ref in self._reference_location_storage]
        return BarsElementTs(bar_positions, color_rgba=(1.0, 1.0, 1.0, 0.5))
