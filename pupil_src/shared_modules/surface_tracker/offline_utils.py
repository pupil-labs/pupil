"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .surface import Surface
from .surface_marker_detector import MarkerDetectorController


class marker_detection_callable(MarkerDetectorController):
    @classmethod
    def from_detector(cls, detector: MarkerDetectorController, marker_min_perimeter):
        return cls(
            marker_detector_mode=detector.marker_detector_mode,
            marker_min_perimeter=marker_min_perimeter,
            square_marker_inverted_markers=detector.inverted_markers,
            square_marker_use_online_mode=False,
            apriltag_nthreads=detector._apriltag_nthreads,
            apriltag_quad_decimate=detector.apriltag_quad_decimate,
            apriltag_decode_sharpening=detector.apriltag_decode_sharpening,
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
        self._context = {}

    def __call__(self, markers):
        markers = {m.uid: m for m in markers}
        return Surface.locate(
            markers,
            self.camera_model,
            self.registered_markers_undist,
            self.registered_markers_dist,
            context=self._context,
        )
