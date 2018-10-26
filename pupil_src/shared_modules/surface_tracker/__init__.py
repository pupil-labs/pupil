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

# TODO The square marker detection should return an object like this already. Also
# this object should offer a mean/centroid function to be used when drawing the
# marker toggle buttons
Square_Marker_Detection = collections.namedtuple(
    "Square_Marker_Detection", ["id", "id_confidence", "verts_px", "perimeter"]
)


class Heatmap_Mode(Enum):
    WITHIN_SURFACE = "Gaze within each surface"
    ACROSS_SURFACES = "Gaze across different surfaces"


class _Surface_Marker_Aggregate(object):
    """
    Stores a list of detections of a specific square marker and aggregates them over
    time to get a more robust localisation.

    A marker detection is represented by the location of the marker vertices. The
    vertices are expected to be in normalized surface coordinates, unlike the
    vertices of a regular Marker, which are located in image pixel space.
    """

    def __init__(self, id, verts_uv=None):
        self.id = id
        self.verts_uv = None
        self.observations = []

        if verts_uv is not None:
            self.verts_uv = np.array(verts_uv)

    def add_observation(self, verts_uv):
        self.observations.append(verts_uv)
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
        self.verts_uv = final_mean

    def save_to_dict(self):
        return {"id": self.id, "verts_uv": [v.tolist() for v in self.verts_uv]}


from surface_tracker.surface_tracker_online import Surface_Tracker_Online
from surface_tracker.surface_tracker_offline import Surface_Tracker_Offline
