"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing

import numpy as np

from .surface_marker import Surface_Marker_UID


class Surface_Marker_Aggregate:
    """
    Stores a list of detections of a specific square marker and aggregates them over
    time to get a more robust localisation.

    A marker detection is represented by the location of the marker vertices. The
    vertices are expected to be in normalized surface coordinates, unlike the
    vertices of a regular Marker, which are located in image pixel space.
    """

    @staticmethod
    def property_equality(
        x: "Surface_Marker_Aggregate", y: "Surface_Marker_Aggregate"
    ) -> bool:
        def property_dict(x: Surface_Marker_Aggregate) -> dict:
            x_dict = x.__dict__.copy()
            x_dict["_verts_uv"] = x_dict["_verts_uv"].tolist()
            return x_dict

        return property_dict(x) == property_dict(y)

    def __init__(
        self, uid: Surface_Marker_UID, verts_uv: typing.Optional[np.ndarray] = None
    ):
        self._uid = uid
        self._verts_uv = None
        self._observations = []

        if verts_uv is not None:
            self._verts_uv = np.asarray(verts_uv)

    def __eq__(self, other):
        return Surface_Marker_Aggregate.property_equality(self, other)

    @property
    def uid(self) -> Surface_Marker_UID:
        return self._uid

    @property
    def verts_uv(self) -> typing.Optional[np.ndarray]:
        return self._verts_uv

    @verts_uv.setter
    def verts_uv(self, new_value: np.ndarray):
        self._verts_uv = new_value

    @property
    def observations(self) -> list:
        return self._observations

    def add_observation(self, verts_uv):
        self._observations.append(verts_uv)
        self._compute_robust_mean()

    def _compute_robust_mean(self):
        # uv is of shape (N, 4, 2) where N is the number of collected observations
        uv = np.asarray(self._observations)
        base_line_mean = np.mean(uv, axis=0)
        distance = np.linalg.norm(uv - base_line_mean, axis=(1, 2))

        # Estimate the mean again using the 50% closest samples
        cut_off = sorted(distance)[len(distance) // 2]
        uv_subset = uv[distance <= cut_off]
        final_mean = np.mean(uv_subset, axis=0)
        self._verts_uv = final_mean
