"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from .surface import Surface
from .surface_marker_detector import MarkerDetectorController


class marker_detection_callable(MarkerDetectorController):
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
