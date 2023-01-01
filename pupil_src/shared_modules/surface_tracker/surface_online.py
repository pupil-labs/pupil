"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections

from .surface import Surface


class Surface_Online(Surface):
    """Surface_Online recalculates it's location on demand."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gaze_history_length = 1
        self.gaze_history = collections.deque()

    def update_location(self, frame_idx, visible_markers, camera_model):
        vis_markers_dict = {m.uid: m for m in visible_markers}

        if not self.defined:
            self._update_definition(frame_idx, vis_markers_dict, camera_model)

        location = self.locate(
            vis_markers_dict,
            camera_model,
            self.registered_markers_undist,
            self.registered_markers_dist,
        )

        self.detected = location.detected
        self.dist_img_to_surf_trans = location.dist_img_to_surf_trans
        self.surf_to_dist_img_trans = location.surf_to_dist_img_trans
        self.img_to_surf_trans = location.img_to_surf_trans
        self.surf_to_img_trans = location.surf_to_img_trans
        self.num_detected_markers = location.num_detected_markers

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
