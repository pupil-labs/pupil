import square_marker_detect
from .surface_tracker import Marker
from .surface import Surface

def marker_detection_callable(min_marker_perimeter, inverted_markers):
    def callable(frame):
        markers = square_marker_detect.detect_markers_robust(
            frame.gray,
            grid_size=5,
            prev_markers=[],
            min_marker_perimeter=min_marker_perimeter,
            aperture=9,
            visualize=0,
            true_detect_every_frame=1,
            invert_image=inverted_markers,
        )
        markers = [
            Marker(m["id"], m["id_confidence"], m["verts"], m["perimeter"])
            for m in markers
        ]
        return markers

    return callable


def surface_locater_callable(camera_model, reg_markers_undist, reg_markers_dist):
    def callable(markers):
        markers = {m.id: m for m in markers}
        return Surface.locate(markers, camera_model, reg_markers_undist, reg_markers_dist)
    return callable