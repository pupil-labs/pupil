
import abc
import typing
import collections

import square_marker_detect
import apriltag

import stdlib_utils




class Surface_Base_Marker(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def from_tuple(state: tuple) -> typing.Optional['Surface_Base_Marker']:
        pass

    @abc.abstractmethod
    def to_tuple(self) -> tuple:
        pass

    @property
    @abc.abstractmethod
    def id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def id_confidence(self) -> float:
        #TODO: Why is it called "id_confidence" instead of "confidence"?
        pass

    @property
    @abc.abstractmethod
    def verts_px(self) -> list:
        pass

    @property
    @abc.abstractmethod
    def perimeter(self) -> float:
        #TODO: Is this used/useful outside surface_marker_detector.py? If not - remove
        pass




# TODO The square marker detection should return an object like this already. Also
# this object should offer a mean/centroid function to be used when drawing the
# marker toggle buttons
class _Square_Marker_Detection(Surface_Base_Marker, collections.namedtuple(
    "Square_Marker_Detection", ["id", "id_confidence", "verts_px", "perimeter"]
)):
    __slots__ = ()

    marker_type = "square" #TODO: Is there a better name?

    @staticmethod
    def from_tuple(state: tuple) -> typing.Optional['_Square_Marker_Detection']:
        if state[-1] != _Square_Marker_Detection.marker_type:
            return None
        state = state[:-1]
        return _Square_Marker_Detection(*state)

    def to_tuple(self) -> tuple:
        state = tuple(self)
        state = (*state, self.marker_type)
        return state



class _Apriltag_V2_Marker_Detection(Surface_Base_Marker, collections.namedtuple(
    "Apriltag_V2_Marker_Detection",
    [
        "tag_family",
        "tag_id",
        "hamming",
        "goodness",
        "decision_margin",
        "homography",
        "center",
        "corners",
    ]
)):
    __slots__ = ()

    marker_type = "apriltag_v2"

    @staticmethod
    def from_tuple(state: tuple) -> typing.Optional['_Apriltag_V2_Marker_Detection']:
        if state[-1] != _Apriltag_V2_Marker_Detection.marker_type:
            return None
        state = state[:-1]
        return _Apriltag_V2_Marker_Detection(*state)

    def to_tuple(self) -> tuple:
        state = tuple(self)
        state = (*state, self.marker_type)
        return state

    @property
    def id(self) -> str:
        return self.tag_id

    @property
    def id_confidence(self) -> float:
        #TODO: Why is it called "id_confidence" instead of "confidence"?
        return 0.95 #FIXME

    @property
    @abc.abstractmethod
    def verts_px(self) -> list:
        # Wrapping each point in a list is needed for compatibility with square detector
        #TODO: See if this wrapping makes sense or if it should be refactored
        return [[point] for point in self.corners]

    @property
    @abc.abstractmethod
    def perimeter(self) -> float:
        #TODO: Is this used/useful outside surface_marker_detector.py? If not - remove
        return 80 #FIXME




class Surface_Marker(Surface_Base_Marker):

    def __getstate__(self) -> tuple:
        return self.to_tuple()

    def __setstate__(self, state: tuple):
        raw_marker = None
        raw_marker = raw_marker or _Square_Marker_Detection.from_tuple(state)
        raw_marker = raw_marker or _Apriltag_V2_Marker_Detection.from_tuple(state)
        raw_marker = raw_marker or _Square_Marker_Detection(*state) #Legacy
        assert raw_marker is not None
        self._raw_marker = raw_marker

    def __init__(self, *args):
        self.__setstate__(tuple(args))

    @staticmethod
    def from_tuple(state: tuple) -> 'Surface_Marker':
        return Surface_Marker(*state)

    def to_tuple(self) -> tuple:
        return self._raw_marker.to_tuple()

    @property
    def id(self) -> str:
        return self._raw_marker.id

    @property
    def id_confidence(self) -> float:
        #TODO: Why is it called "id_confidence" instead of "confidence"?
        return self._raw_marker.id_confidence

    @property
    def verts_px(self) -> list:
        return self._raw_marker.verts_px

    @property
    def perimeter(self) -> float:
        #TODO: Is this used/useful outside surface_marker_detector.py? If not - remove
        return self._raw_marker.perimeter










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

