import square_marker_detect
from .surface import Surface
from .surface_marker_detector import Surface_Marker, Surface_Marker_Detector


class marker_detection_callable(Surface_Marker_Detector):

    def __init__(self, min_marker_perimeter, inverted_markers):
        super().__init__(
            square_marker_min_perimeter=min_marker_perimeter,
            square_marker_inverted_markers=inverted_markers,
        )

    def __call__(self, frame):
        return self.detect_markers(gray_img=frame.gray)


class surface_locater_callable:

    def __init__(
        self, camera_model, registered_markers_undist, registered_markers_dist
    ):
        self.camera_model = camera_model
        self.registered_markers_undist = registered_markers_undist
        self.registered_markers_dist = registered_markers_dist

    def __call__(self, markers):
        markers = {m.id: m for m in markers}
        return Surface.locate(
            markers,
            self.camera_model,
            self.registered_markers_undist,
            self.registered_markers_dist,
        )
