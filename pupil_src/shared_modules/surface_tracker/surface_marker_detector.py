"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import enum
import typing as T
import itertools

import square_marker_detect
import pupil_apriltags

from .surface_marker import Surface_Marker, Surface_Marker_Type


__all__ = [
    "MarkerDetectorController",
    "MarkerDetectorMode",
    "MarkerType",
    "ApriltagFamily",
]


@enum.unique
class MarkerType(enum.Enum):
    SQUARE_MARKER = "square_marker"
    APRILTAG_MARKER = "apriltag_marker"


@enum.unique
class ApriltagFamily(enum.Enum):
    tag16h5 = "tag16h5"
    tag25h9 = "tag25h9"
    tag36h11 = "tag36h11"


class MarkerDetectorMode(T.NamedTuple):
    marker_type: MarkerType
    family: T.Optional[ApriltagFamily] = None

    @classmethod
    def all_supported_cases(cls) -> T.Set["MarkerDetectorMode"]:
        all_square = {cls(MarkerType.SQUARE_MARKER)}
        all_apriltag = {
            cls(MarkerType.APRILTAG_MARKER, family) for family in ApriltagFamily
        }
        return all_square | all_apriltag

    @classmethod
    def from_marker(cls, marker: Surface_Marker) -> "MarkerDetectorMode":
        marker_type = marker.marker_type
        if marker_type == Surface_Marker_Type.SQUARE:
            return cls(MarkerType.SQUARE_MARKER)
        if marker_type == Surface_Marker_Type.APRILTAG_V3:
            return cls(MarkerType.APRILTAG_MARKER, marker.tag_family)
        raise ValueError(
            f"Can't map marker of type '{marker_type}' to a detection mode"
        )

    @property
    def label(self) -> str:
        if self.marker_type == MarkerType.SQUARE_MARKER:
            return "Legacy square markers"
        if self.marker_type == MarkerType.APRILTAG_MARKER:
            return f"Apriltag ({self.family.value})"
        raise ValueError(f"Unlabeled surface marker mode: {self}")

    def as_tuple(self):
        if self.family is not None:
            return (self.marker_type.value, self.family.value)
        else:
            return (self.marker_type.value,)

    @classmethod
    def from_tuple(cls, values: T.Union[T.Tuple[str], T.Tuple[str, str]]):
        marker_type = MarkerType(values[0])
        if marker_type == MarkerType.APRILTAG_MARKER:
            family = ApriltagFamily(values[1])
        else:
            family = None
        return cls(marker_type, family)


class Surface_Base_Marker_Detector(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def inverted_markers(self) -> bool:
        pass

    @inverted_markers.setter
    @abc.abstractmethod
    def inverted_markers(self, value: bool):
        pass

    @property
    @abc.abstractmethod
    def marker_min_perimeter(self) -> int:
        pass

    @marker_min_perimeter.setter
    @abc.abstractmethod
    def marker_min_perimeter(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def marker_detector_modes(self) -> T.Set[MarkerDetectorMode]:
        pass

    @marker_detector_modes.setter
    @abc.abstractmethod
    def marker_detector_modes(self, value: T.Set[MarkerDetectorMode]):
        pass

    @abc.abstractmethod
    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        pass

    def detect_markers(self, gray_img, frame_index: int) -> T.List[Surface_Marker]:
        return list(
            self.detect_markers_iter(gray_img=gray_img, frame_index=frame_index)
        )

    def _surface_marker_filter(self, marker: Surface_Marker) -> bool:
        return self.marker_min_perimeter <= marker.perimeter


class Surface_Square_Marker_Detector(Surface_Base_Marker_Detector):
    def __init__(
        self,
        marker_detector_modes: T.Set[MarkerDetectorMode],
        marker_min_perimeter: int = ...,
        square_marker_inverted_markers: bool = ...,
        square_marker_use_online_mode: bool = ...,
    ):
        self.__marker_min_perimeter = (
            marker_min_perimeter if marker_min_perimeter is not ... else 60
        )
        self.__inverted_markers = (
            square_marker_inverted_markers
            if square_marker_inverted_markers is not ...
            else False
        )
        self.__marker_detector_modes = marker_detector_modes
        self.__previous_raw_markers = []
        self.__previous_frame_index = -1
        self.use_online_mode = (
            square_marker_use_online_mode
            if square_marker_use_online_mode is not ...
            else False
        )

    @property
    def inverted_markers(self) -> bool:
        return self.__inverted_markers

    @inverted_markers.setter
    def inverted_markers(self, value: bool):
        self.__inverted_markers = value

    @property
    def marker_min_perimeter(self) -> int:
        return self.__marker_min_perimeter

    @marker_min_perimeter.setter
    def marker_min_perimeter(self, value: int):
        self.__marker_min_perimeter = value

    @property
    def marker_detector_modes(self) -> T.Set[MarkerDetectorMode]:
        return self.__marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: T.Set[MarkerDetectorMode]):
        self.__marker_detector_modes = value

    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        if MarkerType.SQUARE_MARKER not in (
            mode.marker_type for mode in self.__marker_detector_modes
        ):
            return []

        if self.use_online_mode:
            true_detect_every_frame = 3
        else:
            true_detect_every_frame = 1
            # in offline mode we can get non-monotonic data,
            # in which case the previous data is invalid
            if frame_index != self.__previous_frame_index + 1:
                self.__previous_raw_markers = []
            # TODO: Does this mean that seeking in the recording while the
            # surface is being detected will essentially compromise the data? As
            # in these cases we cannot use the previous frame data for inferring
            # better marker positions. But if we would not have seeked we could
            # have used this information! This looks like an inconsistency!

        grid_size = 5
        aperture = 9
        min_perimeter = self.marker_min_perimeter

        markers = square_marker_detect.detect_markers_robust(
            gray_img=gray_img,
            grid_size=grid_size,
            min_marker_perimeter=min_perimeter,
            aperture=aperture,
            prev_markers=self.__previous_raw_markers,
            true_detect_every_frame=true_detect_every_frame,
            invert_image=self.__inverted_markers,
        )

        # Robust marker detection requires previous markers to be in a different
        # format than the surface tracker.
        self.__previous_raw_markers = markers
        self.__previous_frame_index = frame_index
        markers = map(Surface_Marker.from_square_tag_detection, markers)
        markers = filter(self._surface_marker_filter, markers)
        return markers


class Surface_Apriltag_V3_Marker_Detector_Params:
    def __init__(
        self,
        families: T.Iterable[ApriltagFamily],
        nthreads: int = ...,
        quad_decimate: float = ...,
        quad_sigma: float = ...,
        refine_edges: int = ...,
        decode_sharpening: float = ...,
        debug: bool = ...,
    ):
        assert len(families) > 0
        self.families = families
        self.nthreads = nthreads
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.refine_edges = refine_edges
        self.decode_sharpening = decode_sharpening
        self.debug = debug

    def to_dict(self):
        d = {"families": " ".join(F.value for F in self.families)}
        if self.nthreads is not ...:
            d["nthreads"] = self.nthreads
        if self.quad_decimate is not ...:
            d["quad_decimate"] = self.quad_decimate
        if self.quad_sigma is not ...:
            d["quad_sigma"] = self.quad_sigma
        if self.refine_edges is not ...:
            d["refine_edges"] = self.refine_edges
        if self.decode_sharpening is not ...:
            d["decode_sharpening"] = self.decode_sharpening
        if self.debug is not ...:
            d["debug"] = int(self.debug)
        return d


class Surface_Apriltag_V3_Marker_Detector(Surface_Base_Marker_Detector):
    def __getstate__(self):
        return (
            self.__detector_params,
            self.__marker_min_perimeter,
            self.__marker_detector_modes,
        )

    def __setstate__(self, state):
        (
            self.__detector_params,
            self.__marker_min_perimeter,
            self.__marker_detector_modes,
        ) = state
        params = self.__detector_params.to_dict()
        self._detector = pupil_apriltags.Detector(**params)

    def __init__(
        self,
        marker_detector_modes: T.Set[MarkerDetectorMode],
        marker_min_perimeter: int = ...,
        apriltag_families: T.Set[ApriltagFamily] = ...,
        apriltag_nthreads: int = ...,
        apriltag_quad_decimate: float = ...,
        apriltag_quad_sigma: float = ...,
        apriltag_refine_edges: bool = ...,
        apriltag_decode_sharpening: float = ...,
        apriltag_debug: bool = ...,
    ):
        detector_params = Surface_Apriltag_V3_Marker_Detector_Params(
            families=apriltag_families,
            nthreads=apriltag_nthreads,
            quad_decimate=apriltag_quad_decimate,
            quad_sigma=apriltag_quad_sigma,
            refine_edges=apriltag_refine_edges,
            decode_sharpening=apriltag_decode_sharpening,
            debug=apriltag_debug,
        )
        state = (detector_params, marker_min_perimeter, marker_detector_modes)
        self.__setstate__(state)

    @property
    def inverted_markers(self) -> bool:
        return False

    @inverted_markers.setter
    def inverted_markers(self, value: bool):
        pass  # nop

    @property
    def marker_min_perimeter(self) -> int:
        return self.__marker_min_perimeter

    @marker_min_perimeter.setter
    def marker_min_perimeter(self, value: int):
        self.__marker_min_perimeter = value

    @property
    def marker_detector_modes(self) -> T.Set[MarkerDetectorMode]:
        return self.__marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: T.Set[MarkerDetectorMode]):
        self.__marker_detector_modes = value

    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        if MarkerType.APRILTAG_MARKER not in (
            mode.marker_type for mode in self.__marker_detector_modes
        ):
            return []
        markers = self._detector.detect(img=gray_img)
        markers = map(Surface_Marker.from_apriltag_v3_detection, markers)
        markers = filter(self._surface_marker_filter, markers)
        return markers


class MarkerDetectorController(Surface_Base_Marker_Detector):
    def __init__(
        self,
        marker_detector_modes: T.Set[MarkerDetectorMode],
        marker_min_perimeter: int = ...,
        square_marker_inverted_markers: bool = ...,
        square_marker_use_online_mode: bool = ...,
        apriltag_nthreads: int = ...,
        apriltag_quad_decimate: float = ...,
        apriltag_decode_sharpening: float = ...,
    ):
        self.__square_detector = Surface_Square_Marker_Detector(
            marker_detector_modes=marker_detector_modes,
            marker_min_perimeter=marker_min_perimeter,
            square_marker_inverted_markers=square_marker_inverted_markers,
            square_marker_use_online_mode=square_marker_use_online_mode,
        )
        families = {
            mode.family
            for mode in marker_detector_modes
            if mode.marker_type == MarkerType.APRILTAG_MARKER
        }
        self.__apriltag_detector = Surface_Apriltag_V3_Marker_Detector(
            marker_detector_modes=marker_detector_modes,
            marker_min_perimeter=marker_min_perimeter,
            apriltag_families=families,
            apriltag_nthreads=apriltag_nthreads,
            apriltag_quad_decimate=apriltag_quad_decimate,
            apriltag_decode_sharpening=apriltag_decode_sharpening,
        )

    @property
    def inverted_markers(self) -> bool:
        return self.__square_detector.inverted_markers

    @inverted_markers.setter
    def inverted_markers(self, value: bool):
        self.__square_detector.inverted_markers = value

    @property
    def marker_min_perimeter(self) -> int:
        min_perimeter = self.__apriltag_detector.marker_min_perimeter
        assert min_perimeter == self.__square_detector.marker_min_perimeter
        return min_perimeter

    @marker_min_perimeter.setter
    def marker_min_perimeter(self, value: int):
        self.__square_detector.marker_min_perimeter = value
        self.__apriltag_detector.marker_min_perimeter = value

    @property
    def marker_detector_modes(self) -> T.Set[MarkerDetectorMode]:
        marker_detector_modes = self.__apriltag_detector.marker_detector_modes
        assert marker_detector_modes == self.__square_detector.marker_detector_modes
        return marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: T.Set[MarkerDetectorMode]):
        self.__square_detector.marker_detector_modes = value
        self.__apriltag_detector.marker_detector_modes = value

    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        return itertools.chain(
            self.__square_detector.detect_markers_iter(
                gray_img=gray_img, frame_index=frame_index
            ),
            self.__apriltag_detector.detect_markers_iter(
                gray_img=gray_img, frame_index=frame_index
            ),
        )
