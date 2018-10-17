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
from enum import Enum

import numpy as np

Marker = collections.namedtuple("Marker", ["id", "id_confidence", "verts", "perimeter"])


class Heatmap_Mode(Enum):
    WITHIN_SURFACE = "Gaze within each surface"
    ACROSS_SURFACES = "Gaze across different surfaces"


class _Surface_Marker(object):
    """
    A Surface Marker is located in normalized surface space, unlike regular Markers which are
    located in image space. It's location on the surface is aggregated over a list of
    obersvations.
    """

    def __init__(self, id, verts=None):
        self.id = id
        self.verts = None
        self.observations = []

        if not verts is None:
            self.verts = np.array(verts)

    def add_observation(self, uv_coords):
        self.observations.append(uv_coords)
        self._compute_robust_mean()

    def _compute_robust_mean(self):
        # uv is of shape (N, 4, 2) where N is the number of collected observations
        uv = np.array(self.observations)
        base_line_mean = np.mean(uv, axis=0)
        distance = np.linalg.norm(uv - base_line_mean, axis=(1, 2))

        # Estimate the mean again using the 50% closest samples
        cut_off = sorted(distance)[len(distance) // 2]
        uv_subset = uv[distance <= cut_off]
        final_mean = np.mean(uv_subset, axis=0)
        self.verts = final_mean

    def save_to_dict(self):
        return {"id": self.id, "verts": [v.tolist() for v in self.verts]}


from surface_tracker.surface_tracker_online import Surface_Tracker_Online
from surface_tracker.surface_tracker_offline import Surface_Tracker_Offline
