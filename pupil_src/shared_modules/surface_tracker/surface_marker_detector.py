"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import enum
import logging
import typing as T

import pupil_apriltags
import square_marker_detect

from .surface_marker import Surface_Marker, Surface_Marker_Type

logger = logging.getLogger(__name__)

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
    tag25h9 = "tag25h9"
    tag36h11 = "tag36h11"
    tagCircle21h7 = "tagCircle21h7"
    tagCircle49h12 = "tagCircle49h12"
    tagCustom48h12 = "tagCustom48h12"
    tagStandard41h12 = "tagStandard41h12"
    tagStandard52h13 = "tagStandard52h13"


class MarkerDetectorMode(T.NamedTuple):
    marker_type: Surface_Marker_Type
    family: T.Optional[str]

    @classmethod
    def all_supported_cases(cls) -> T.Set["MarkerDetectorMode"]:
        all_square = {cls(MarkerType.SQUARE_MARKER, None)}
        all_apriltag = {
            cls(MarkerType.APRILTAG_MARKER, family) for family in ApriltagFamily
        }
        return all_square | all_apriltag

    @classmethod
    def from_marker(cls, marker: Surface_Marker) -> "MarkerDetectorMode":
        marker_type = marker.marker_type
        if marker_type == Surface_Marker_Type.SQUARE:
            return cls(MarkerType.SQUARE_MARKER, None)
        if marker_type == Surface_Marker_Type.APRILTAG_V3:
            return cls(
                MarkerType.APRILTAG_MARKER, ApriltagFamily(marker.raw_marker.tag_family)
            )
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
            return (self.marker_type.value, None)

    @classmethod
    def from_tuple(cls, values: T.Union[T.Tuple[str], T.Tuple[str, str]]):
        marker_type = MarkerType(values[0])
        if marker_type == MarkerType.APRILTAG_MARKER:
            family = ApriltagFamily(values[1])
        else:
            family = None
        return cls(marker_type, family)


class Surface_Base_Marker_Detector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        pass

    def detect_markers(self, gray_img, frame_index: int) -> T.List[Surface_Marker]:
        return list(
            self.detect_markers_iter(gray_img=gray_img, frame_index=frame_index)
        )


class Surface_Square_Marker_Detector(Surface_Base_Marker_Detector):
    def __init__(
        self,
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

    def _surface_marker_filter(self, marker: Surface_Marker) -> bool:
        return self.marker_min_perimeter <= marker.perimeter

    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
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
        return self.__detector_params

    def __setstate__(self, state):
        self.__detector_params = state
        params = self.__detector_params.to_dict()
        self._detector = pupil_apriltags.Detector(**params)

    def __init__(
        self,
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
        self.__setstate__(detector_params)

    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        markers = self._detector.detect(img=gray_img)
        markers = map(Surface_Marker.from_apriltag_v3_detection, markers)
        return markers


class MarkerDetectorController(Surface_Base_Marker_Detector):
    def __init__(
        self,
        marker_detector_mode: MarkerDetectorMode,
        marker_min_perimeter: int = ...,
        square_marker_inverted_markers: bool = ...,
        square_marker_use_online_mode: bool = ...,
        apriltag_nthreads: int = 2,
        apriltag_quad_decimate: float = ...,
        apriltag_decode_sharpening: float = ...,
    ):
        self._marker_detector_mode = marker_detector_mode
        self._marker_min_perimeter = marker_min_perimeter

        self._square_marker_inverted_markers = square_marker_inverted_markers
        self._square_marker_use_online_mode = square_marker_use_online_mode

        self._apriltag_nthreads = apriltag_nthreads
        self._apriltag_quad_decimate = apriltag_quad_decimate
        self._apriltag_decode_sharpening = apriltag_decode_sharpening

        self.init_detector()

    def init_detector(self):
        if self._marker_detector_mode.marker_type == MarkerType.APRILTAG_MARKER:
            self.__detector = Surface_Apriltag_V3_Marker_Detector(
                apriltag_families={self._marker_detector_mode.family},
                apriltag_nthreads=self._apriltag_nthreads,
                apriltag_quad_decimate=self._apriltag_quad_decimate,
                apriltag_decode_sharpening=self._apriltag_decode_sharpening,
            )
            logger.debug(
                "Init Apriltag Detector (\n"
                f"\tapriltag_families={self._marker_detector_mode.family}\n"
                f"\tapriltag_nthreads={self._apriltag_nthreads}\n"
                f"\tapriltag_quad_decimate={self._apriltag_quad_decimate}\n"
                f"\tapriltag_decode_sharpening={self._apriltag_decode_sharpening}\n"
                ")"
            )
        elif self._marker_detector_mode.marker_type == MarkerType.SQUARE_MARKER:
            self.__detector = Surface_Square_Marker_Detector(
                marker_min_perimeter=self._marker_min_perimeter,
                square_marker_inverted_markers=self._square_marker_inverted_markers,
                square_marker_use_online_mode=self._square_marker_use_online_mode,
            )

    @property
    def inverted_markers(self) -> bool:
        return self._square_marker_inverted_markers

    @inverted_markers.setter
    def inverted_markers(self, value: bool):
        self._square_marker_inverted_markers = value
        if self.marker_detector_mode.marker_type == MarkerType.SQUARE_MARKER:
            self.init_detector()

    @property
    def marker_min_perimeter(self) -> int:
        return self._marker_min_perimeter

    @marker_min_perimeter.setter
    def marker_min_perimeter(self, value: int):
        self._marker_min_perimeter = value
        if self.marker_detector_mode.marker_type == MarkerType.SQUARE_MARKER:
            self.init_detector()

    @property
    def marker_detector_mode(self) -> MarkerDetectorMode:
        return self._marker_detector_mode

    @marker_detector_mode.setter
    def marker_detector_mode(self, value: MarkerDetectorMode):
        self._marker_detector_mode = value
        self.init_detector()

    @property
    def apriltag_quad_decimate(self) -> float:
        return self._apriltag_quad_decimate

    @apriltag_quad_decimate.setter
    def apriltag_quad_decimate(self, value: float):
        self._apriltag_quad_decimate = value
        if self.marker_detector_mode.marker_type == MarkerType.APRILTAG_MARKER:
            self.init_detector()

    @property
    def apriltag_decode_sharpening(self) -> float:
        return self._apriltag_decode_sharpening

    @apriltag_decode_sharpening.setter
    def apriltag_decode_sharpening(self, value: float):
        self._apriltag_decode_sharpening = value
        if self.marker_detector_mode.marker_type == MarkerType.APRILTAG_MARKER:
            self.init_detector()

    def detect_markers_iter(
        self, gray_img, frame_index: int
    ) -> T.Iterable[Surface_Marker]:
        yield from self.__detector.detect_markers_iter(
            gray_img=gray_img, frame_index=frame_index
        )
