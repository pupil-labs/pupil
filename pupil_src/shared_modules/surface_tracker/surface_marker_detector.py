
import abc
import square_marker_detect
from surface_tracker import Square_Marker_Detection
import stdlib_utils


class Surface_Base_Marker_Detector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def detect_markers(self, gray_img):
        #TODO: Add type hints
        pass


class Surface_Square_Marker_Detector(Surface_Base_Marker_Detector):

    def __init__(self, marker_min_confidence=None, marker_min_perimeter=None, robust_detection=None, inverted_markers=None):
        def param(x, default):
            return x if x is not ... else default
        #
        self.marker_min_perimeter  = param(marker_min_perimeter, 60)
        self.marker_min_confidence = param(marker_min_confidence, 0.1)
        #
        self.robust_detection = param(robust_detection, True)
        self.inverted_markers = param(inverted_markers, False)
        #
        self.previous_raw_markers = []
        self.previous_square_markers_unfiltered = []

    def detect_markers(self, gray_img):
        #TODO: Add type hints

        grid_size = 5
        aperture = 11

        if self.robust_detection:
            markers = square_marker_detect.detect_markers_robust(
                gray_img=gray_img,
                grid_size=grid_size,
                aperture=aperture,
                prev_markers=self.previous_raw_markers,
                true_detect_every_frame=3,
                min_marker_perimeter=self.marker_min_perimeter,
                invert_image=self.inverted_markers,
            )
        else:
            markers = square_marker_detect.detect_markers(
                gray_img=gray_img,
                grid_size=grid_size,
                aperture=aperture,
                min_marker_perimeter=self.marker_min_perimeter,
            )

        # Robust marker detection requires previous markers to be in a different
        # format than the surface tracker.
        self.previous_raw_markers = markers
        markers = [
            Square_Marker_Detection(
                m["id"], m["id_confidence"], m["verts"], m["perimeter"]
            )
            for m in markers
        ]
        markers = self._unique_markers(markers)
        self.previous_square_markers_unfiltered = markers
        markers = self._filter_markers(markers)
        return markers


    def _unique_markers(self, markers):
        #TODO: Add type hints

        # if an id shows twice use the bigger marker (usually this is a screen camera
        # echo artifact.)
        markers = stdlib_utils.unique(
            markers,
            key=lambda m: m.id,
            select=lambda x, y: x if x.perimeter >= y.perimeter else y
        )
        markers = list(markers)
        return markers

    def _filter_markers(self, markers):
        #TODO: Add type hints
        markers = [
            m
            for m in markers
            if m.perimeter >= self.marker_min_perimeter
            and m.id_confidence > self.marker_min_confidence
        ]
        return markers


class Surface_Apriltag_Marker_Detector(Surface_Base_Marker_Detector):
    #TODO: Implement
    pass


class Surface_Combined_Marker_Detector(Surface_Base_Marker_Detector):
    #TODO: Implement
    def __init__(self):
        self._square_detector = Surface_Square_Marker_Detector()
        self._apriltag_detector = Surface_Apriltag_Marker_Detector()

