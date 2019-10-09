"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import typing

from .surface import Surface
from .surface_marker_detector import (
    Surface_Marker_Detector,
    Surface_Marker_Detector_Mode,
)


class marker_detection_callable(Surface_Marker_Detector):
    def __init__(
        self,
        marker_detector_modes: typing.Set[Surface_Marker_Detector_Mode],
        min_marker_perimeter: int,
        inverted_markers: bool,
    ):
        super().__init__(
            marker_detector_modes=marker_detector_modes,
            marker_min_perimeter=min_marker_perimeter,
            square_marker_inverted_markers=inverted_markers,
        )

    def __call__(self, frame):
        return self.detect_markers(gray_img=frame.gray, frame_index=frame.index)


class surface_locater_callable:
    def __init__(
        self, camera_model, registered_markers_undist, registered_markers_dist
    ):
        self.camera_model = camera_model
        self.registered_markers_undist = registered_markers_undist
        self.registered_markers_dist = registered_markers_dist

    def __call__(self, markers):
        markers = {m.uid: m for m in markers}
        return Surface.locate(
            markers,
            self.camera_model,
            self.registered_markers_undist,
            self.registered_markers_dist,
        )


class surfaces_locator_callable:
    def __init__(self, surfaces, camera_model):
        self.camera_model = camera_model
        surfaces_registered_markers_dist = [s.registered_markers_dist for s in surfaces]
        surfaces_registered_markers_undist = [
            s.registered_markers_undist for s in surfaces
        ]
        self.surfaces_data = list(
            zip(surfaces_registered_markers_dist, surfaces_registered_markers_undist)
        )

    def __call__(self, markers):
        markers = {m.uid: m for m in markers}

        def locate_surface(surface_data):
            registered_markers_dist, registered_markers_undist = surface_data
            return Surface.locate(
                markers,
                self.camera_model,
                registered_markers_undist,
                registered_markers_dist,
            )

        return list(enumerate(map(locate_surface, self.surfaces_data)))
