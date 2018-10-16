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

import cv2
import numpy as np

from surface_tracker_future.surface import Surface


class Surface_Online(Surface):
    def __init__(self, init_dict=None):
        super().__init__(init_dict=init_dict)

        self.gaze_history_length = 1
        self.gaze_history = collections.deque()

    def update_location(self, idx, vis_markers, camera_model):
        vis_markers_dict = {m.id: m for m in vis_markers}

        if not self.defined:
            self.update_def(idx, vis_markers_dict, camera_model)

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
        for event in gaze_on_surf:
            if (
                event["confidence"] < self.heatmap_min_data_confidence
                and event["on_surf"]
            ):
                continue
            self.gaze_history.append(
                {"timestamp": event["timestamp"], "gaze": event["norm_pos"]}
            )

    def update_heatmap(self):
        data = [x["gaze"] for x in self.gaze_history]
        self._generate_heatmap(data)
