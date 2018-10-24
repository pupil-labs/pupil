"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections

from surface_tracker.surface import Surface


class Surface_Online(Surface):
    """Surface_Online recalculates it's location on demand."""

    def __init__(self, name="unknown", init_dict=None):
        super().__init__(name=name, init_dict=init_dict)

        self.gaze_history_length = 1
        self.gaze_history = collections.deque()

    def update_location(self, frame_idx, visible_markers, camera_model):
        vis_markers_dict = {m.id: m for m in visible_markers}

        if not self.defined:
            self._update_definition(frame_idx, vis_markers_dict, camera_model)

        # Get dict of current transformations
        transformations = self.locate(
            vis_markers_dict,
            camera_model,
            self.reg_markers_undist,
            self.reg_markers_dist,
        )
        self.__dict__.update(transformations)

    def update_gaze_history(self, gaze_on_surf, world_timestamp):
        # Remove old entries
        while (
            self.gaze_history
            and world_timestamp - self.gaze_history[0]["timestamp"]
            > self.gaze_history_length
        ):
            self.gaze_history.popleft()

        # Add new entries
        self.gaze_history += gaze_on_surf
