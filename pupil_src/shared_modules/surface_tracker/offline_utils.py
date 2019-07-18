import square_marker_detect
from .surface import Surface
from .surface_tracker import Square_Marker_Detection


class marker_detection_callable:
    def __init__(self, min_marker_perimeter, inverted_markers):
        self.min_marker_perimeter = min_marker_perimeter
        self.inverted_markers = inverted_markers
        self.prev_markers = []
        self.prev_frame_idx = -1

    def __call__(self, frame):
        if frame.index != self.prev_frame_idx + 1:
            self.prev_markers = []

        markers = square_marker_detect.detect_markers_robust(
            frame.gray,
            grid_size=5,
            prev_markers=self.prev_markers,
            min_marker_perimeter=self.min_marker_perimeter,
            aperture=9,
            visualize=0,
            true_detect_every_frame=1,
            invert_image=self.inverted_markers,
        )

        self.prev_markers = markers
        self.prev_frame_idx = frame.index

        markers = [
            Square_Marker_Detection(
                m["id"], m["id_confidence"], m["verts"], m["perimeter"]
            )
            for m in markers
        ]
        return markers


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
