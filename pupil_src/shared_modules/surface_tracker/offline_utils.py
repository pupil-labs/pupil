import square_marker_detect
from .surface import Surface
from .surface_marker_detector import Surface_Marker


# TODO: Refactor clients of this to use Surface_Marker_Detector instances
class marker_detection_callable:
    def __init__(self, min_marker_perimeter, inverted_markers):
        self.min_marker_perimeter = min_marker_perimeter
        self.inverted_markers = inverted_markers

    def __call__(self, frame):
        markers = square_marker_detect.detect_markers_robust(
            frame.gray,
            grid_size=5,
            prev_markers=[],
            min_marker_perimeter=self.min_marker_perimeter,
            aperture=9,
            visualize=0,
            true_detect_every_frame=1,
            invert_image=self.inverted_markers,
        )
        return [ Surface_Marker.from_square_tag_detection(m) for m in markers]


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
