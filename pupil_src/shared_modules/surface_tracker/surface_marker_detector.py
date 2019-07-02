
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

_Raw_Square_Marker_Detection = collections.namedtuple(
    "Square_Marker_Detection",
    [
        "id",
        "id_confidence",
        "verts_px",
        "perimeter",
    ]
)

class _Square_Marker_Detection(_Raw_Square_Marker_Detection, Surface_Base_Marker):
    __slots__ = ()

    marker_type = "square" #TODO: Is there a better name?

    @staticmethod
    def from_tuple(state: tuple) -> typing.Optional['_Square_Marker_Detection']:
        if state[-1] != _Square_Marker_Detection.marker_type:
            return None
        return _Square_Marker_Detection(*state[:-1])

    def to_tuple(self) -> tuple:
        state = tuple(self)
        state = (*state, self.marker_type)
        return state


_Raw_Apriltag_V2_Marker_Detection = collections.namedtuple(
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
)

class _Apriltag_V2_Marker_Detection(_Raw_Apriltag_V2_Marker_Detection, Surface_Base_Marker):
    __slots__ = ()

    marker_type = "apriltag_v2"

    @staticmethod
    def from_tuple(state: tuple) -> typing.Optional['_Apriltag_V2_Marker_Detection']:
        if state[-1] != _Apriltag_V2_Marker_Detection.marker_type:
            return None
        return _Apriltag_V2_Marker_Detection(*state[:-1])

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
    def detect_markers(self, gray_img) -> typing.List[Surface_Marker]:
        #TODO: Add type hints
        pass



class Surface_Square_Marker_Detector(Surface_Base_Marker_Detector):

    def __init__(
        self,
        square_marker_min_confidence: float=...,
        square_marker_min_perimeter: int=...,
        square_marker_robust_detection: bool=...,
        square_marker_inverted_markers: bool=...,
        **kwargs,
    ):
        Param_T = typing.TypeVar('Param_T')
        def param(x: Param_T, default: Param_T) -> Param_T:
            return x if x is not ... else default
        #
        self.marker_min_perimeter  = param(square_marker_min_perimeter, 60)
        self.marker_min_confidence = param(square_marker_min_confidence, 0.1)
        #
        self.robust_detection = param(square_marker_robust_detection, True)
        self.inverted_markers = param(square_marker_inverted_markers, False)
        #
        self.previous_raw_markers = []
        self.previous_square_markers_unfiltered = []

    @property
    def markers_unfiltered(self):
        return self.previous_square_markers_unfiltered

    def detect_markers(self, gray_img) -> typing.List[Surface_Marker]:
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
        markers = map(self._marker_from_raw, markers)
        markers = self._unique_markers(markers)
        self.previous_square_markers_unfiltered = markers
        markers = self._filter_markers(markers)
        return markers

    def _marker_from_raw(self, raw_marker: dict) -> Surface_Marker:
        square_marker = _Square_Marker_Detection(
            id=raw_marker["id"],
            id_confidence=raw_marker["id_confidence"],
            verts_px=raw_marker["verts"],
            perimeter=raw_marker["perimeter"],
        )
        return Surface_Marker.from_tuple(square_marker.to_tuple())

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

    @staticmethod
    def _detector_options(
        families: str=...,
        border: int=...,
        nthreads: int=...,
        quad_decimate: float=...,
        quad_blur: float=...,
        refine_edges: bool=...,
        refine_decode: bool=...,
        refine_pose: bool=...,
        debug: bool=...,
        quad_contours: bool=...,
    ):
        options = apriltag.DetectorOptions()
        if families is not ...:
            options.families = str(families)
        if border is not ...:
            options.border = int(border)
        if nthreads is not ...:
            options.nthreads = int(nthreads)
        if quad_decimate is not ...:
            options.quad_decimate = float(quad_decimate)
        if quad_blur is not ...:
            options.quad_sigma = float(quad_blur)
        if refine_edges is not ...:
            options.refine_edges = int(refine_edges)
        if refine_decode is not ...:
            options.refine_decode = int(refine_decode)
        if refine_pose is not ...:
            options.refine_pose = int(refine_pose)
        if debug is not ...:
            options.debug = int(debug)
        if quad_contours is not ...:
            options.quad_contours = bool(quad_contours)
        return options

    def __init__(
        self,
        apriltag_families: str=...,
        apriltag_border: int=...,
        apriltag_nthreads: int=...,
        apriltag_quad_decimate: float=...,
        apriltag_quad_blur: float=...,
        apriltag_refine_edges: bool=...,
        apriltag_refine_decode: bool=...,
        apriltag_refine_pose: bool=...,
        apriltag_debug: bool=...,
        apriltag_quad_contours: bool=...,
        **kwargs,
    ):
        options = type(self)._detector_options(
            families=apriltag_families,
            border=apriltag_border,
            nthreads=apriltag_nthreads,
            quad_decimate=apriltag_quad_decimate,
            quad_blur=apriltag_quad_blur,
            refine_edges=apriltag_refine_edges,
            refine_decode=apriltag_refine_decode,
            refine_pose=apriltag_refine_pose,
            debug=apriltag_debug,
            quad_contours=apriltag_quad_contours,
        )
        self._detector = apriltag.Detector(
            detector_options=options,
        )

        #TODO: Remove these external dependencies
        self.marker_min_perimeter  = 60
        self.marker_min_confidence = 0.1
        self.robust_detection = True
        self.inverted_markers = False
        self.previous_raw_markers = []
        self.previous_apriltag_markers_unfiltered = []

    @property
    def markers_unfiltered(self):
        return self.previous_apriltag_markers_unfiltered

    def detect_markers(self, gray_img) -> typing.List[Surface_Marker]:
        markers = self._detector.detect(img=gray_img)
        markers = map(self._marker_from_raw, markers)
        markers = list(markers)
        self.previous_apriltag_markers_unfiltered = markers
        return markers

    def _marker_from_raw(self, raw_marker: apriltag.DetectionBase) -> Surface_Marker:
        apriltag_marker = _Apriltag_V2_Marker_Detection(
            tag_family=raw_marker.tag_family,
            tag_id=raw_marker.tag_id,
            hamming=raw_marker.hamming,
            goodness=raw_marker.goodness,
            decision_margin=raw_marker.decision_margin,
            homography=raw_marker.homography,
            center=raw_marker.center,
            corners=raw_marker.corners,
        )
        return Surface_Marker.from_tuple(apriltag_marker.to_tuple())

    def _unique_markers(self, markers):
        return markers #FIXME: Remove dependency on this method

    def _filter_markers(self, markers):
        return markers #FIXME: Remove dependency on this method



class Surface_Combined_Marker_Detector(Surface_Base_Marker_Detector):
    #TODO: Implement
    def __init__(
        self,
        square_marker_min_confidence: float=...,
        square_marker_min_perimeter: int=...,
        square_marker_robust_detection: bool=...,
        square_marker_inverted_markers: bool=...,
        apriltag_families: str=...,
        apriltag_border: int=...,
        apriltag_nthreads: int=...,
        apriltag_quad_decimate: float=...,
        apriltag_quad_blur: float=...,
        apriltag_refine_edges: bool=...,
        apriltag_refine_decode: bool=...,
        apriltag_refine_pose: bool=...,
        apriltag_debug: bool=...,
        apriltag_quad_contours: bool=...,
    ):
        self._square_detector = Surface_Square_Marker_Detector(
            marker_min_confidence=square_marker_min_confidence,
            marker_min_perimeter=square_marker_min_perimeter,
            robust_detection=square_marker_robust_detection,
            inverted_markers=square_marker_inverted_markers,
        )
        self._apriltag_detector = Surface_Apriltag_Marker_Detector(
            families=apriltag_families,
            border=apriltag_border,
            nthreads=apriltag_nthreads,
            quad_decimate=apriltag_quad_decimate,
            quad_blur=apriltag_quad_blur,
            refine_edges=apriltag_refine_edges,
            refine_decode=apriltag_refine_decode,
            refine_pose=apriltag_refine_pose,
            debug=apriltag_debug,
            quad_contours=apriltag_quad_contours,
        )

    def detect_markers(self, gray_img) -> typing.List[Surface_Marker]:
        markers = []
        markers += self._square_detector.detect_markers(gray_img=gray_img)
        markers += self._apriltag_detector.detect_markers(gray_img=gray_img)
        return markers
