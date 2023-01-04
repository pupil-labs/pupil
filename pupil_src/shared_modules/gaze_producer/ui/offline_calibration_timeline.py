"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class OfflineCalibrationTimeline:
    def __init__(
        self, plugin_timeline, reference_location_timeline, gaze_mapper_timeline, plugin
    ):
        self._plugin_timeline = plugin_timeline
        self._reference_location_timeline = reference_location_timeline
        self._gaze_mapper_timeline = gaze_mapper_timeline

        reference_location_timeline.render_parent_timeline = self.render
        gaze_mapper_timeline.render_parent_timeline = self.render

        plugin.add_observer("init_ui", self._on_init_ui)

    def render(self):
        self._plugin_timeline.clear_rows()
        self._plugin_timeline.add_row(self._reference_location_timeline.create_row())
        for row in self._gaze_mapper_timeline.create_rows():
            self._plugin_timeline.add_row(row)
        self._plugin_timeline.refresh()

    def _on_init_ui(self):
        self.render()
