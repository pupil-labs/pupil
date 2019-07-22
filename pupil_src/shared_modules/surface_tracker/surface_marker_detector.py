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
import typing
import itertools

import square_marker_detect
import apriltag

from .surface_marker import Surface_Marker, Surface_Marker_Type


__all__ = ["Surface_Marker_Detector", "Surface_Marker_Detector_Mode"]


DEFAULT_APRILTAG_FAMILY = "tag36h11"


@enum.unique
class Surface_Marker_Detector_Mode(enum.Enum):
    SQUARE_MARKER = 'square_marker'
    APRILTAG_MARKER = 'apriltag_marker'

    @staticmethod
    def all_supported_cases() -> typing.Set["Surface_Marker_Detector_Mode"]:
        return set(Surface_Marker_Detector_Mode)

    @staticmethod
    def from_marker(marker: Surface_Marker) -> "Surface_Marker_Detector_Mode":
        marker_type = marker.marker_type
        if marker_type == Surface_Marker_Type.SQUARE:
            return Surface_Marker_Detector_Mode.SQUARE_MARKER
        if marker_type == Surface_Marker_Type.APRILTAG_V2:
            return Surface_Marker_Detector_Mode.APRILTAG_MARKER
        raise ValueError(f"Can't map marker of type '{marker_type}' to a detection mode")

    @property
    def label(self) -> str:
        if self == Surface_Marker_Detector_Mode.SQUARE_MARKER:
            return "Legacy square markers"
        if self == Surface_Marker_Detector_Mode.APRILTAG_MARKER:
            return f"Apriltag ({DEFAULT_APRILTAG_FAMILY})"
        raise ValueError(f"Unlabeled surface marker mode: {self}")


class Surface_Base_Marker_Detector(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def robust_detection(self) -> bool:
        pass

    @robust_detection.setter
    @abc.abstractmethod
    def robust_detection(self, value: bool):
        pass

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
    def marker_detector_modes(self) -> typing.Set[Surface_Marker_Detector_Mode]:
        pass

    @marker_detector_modes.setter
    @abc.abstractmethod
    def marker_detector_modes(self, value: typing.Set[Surface_Marker_Detector_Mode]):
        pass

    @abc.abstractmethod
    def detect_markers_iter(self, gray_img) -> typing.Iterable[Surface_Marker]:
        pass

    def detect_markers(self, gray_img) -> typing.List[Surface_Marker]:
        return list(self.detect_markers_iter(gray_img=gray_img))

    def _surface_marker_filter(self, marker: Surface_Marker) -> bool:
        return self.marker_min_perimeter <= marker.perimeter


class Surface_Square_Marker_Detector(Surface_Base_Marker_Detector):
    def __init__(
        self,
        marker_detector_modes: typing.Set[Surface_Marker_Detector_Mode],
        marker_min_perimeter: int = ...,
        square_marker_robust_detection: bool = ...,
        square_marker_inverted_markers: bool = ...,
        **kwargs,
    ):
        self.__marker_min_perimeter = (
            marker_min_perimeter
            if marker_min_perimeter is not ...
            else 60
        )
        self.__robust_detection = (
            square_marker_robust_detection
            if square_marker_robust_detection is not ...
            else True
        )
        self.__inverted_markers = (
            square_marker_inverted_markers
            if square_marker_inverted_markers is not ...
            else False
        )
        self.__marker_detector_modes = marker_detector_modes
        self.__previous_raw_markers = []

    @property
    def robust_detection(self) -> bool:
        return self.__robust_detection

    @robust_detection.setter
    def robust_detection(self, value: bool):
        self.__robust_detection = value

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
    def marker_detector_modes(self) -> typing.Set[Surface_Marker_Detector_Mode]:
        return self.__marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: typing.Set[Surface_Marker_Detector_Mode]):
        self.__marker_detector_modes = value

    def detect_markers_iter(self, gray_img) -> typing.Iterable[Surface_Marker]:
        if Surface_Marker_Detector_Mode.SQUARE_MARKER not in self.marker_detector_modes:
            return []

        grid_size = 5
        aperture = 11
        min_perimeter = self.marker_min_perimeter

        if self.__robust_detection:
            markers = square_marker_detect.detect_markers_robust(
                gray_img=gray_img,
                grid_size=grid_size,
                min_marker_perimeter=min_perimeter,
                aperture=aperture,
                prev_markers=self.__previous_raw_markers,
                true_detect_every_frame=3,
                invert_image=self.__inverted_markers,
            )
        else:
            markers = square_marker_detect.detect_markers(
                gray_img=gray_img,
                grid_size=grid_size,
                min_marker_perimeter=min_perimeter,
                aperture=aperture,
            )

        # Robust marker detection requires previous markers to be in a different
        # format than the surface tracker.
        self.__previous_raw_markers = markers
        markers = map(Surface_Marker.from_square_tag_detection, markers)
        markers = filter(self._surface_marker_filter, markers)
        return markers


class _Apriltag_V2_Detector_Options(apriltag.DetectorOptions):
    def __init__(
        self,
        families: str = ...,
        border: int = ...,
        nthreads: int = ...,
        quad_decimate: float = ...,
        quad_blur: float = ...,
        refine_edges: bool = ...,
        refine_decode: bool = ...,
        refine_pose: bool = ...,
        debug: bool = ...,
        quad_contours: bool = ...,
    ):
        super().__init__()
        if families is not ...:
            self.families = str(families)
        if border is not ...:
            self.border = int(border)
        if nthreads is not ...:
            self.nthreads = int(nthreads)
        if quad_decimate is not ...:
            self.quad_decimate = float(quad_decimate)
        if quad_blur is not ...:
            self.quad_sigma = float(quad_blur)
        if refine_edges is not ...:
            self.refine_edges = int(refine_edges)
        if refine_decode is not ...:
            self.refine_decode = int(refine_decode)
        if refine_pose is not ...:
            self.refine_pose = int(refine_pose)
        if debug is not ...:
            self.debug = int(debug)
        if quad_contours is not ...:
            self.quad_contours = bool(quad_contours)


class Surface_Apriltag_V2_Marker_Detector(Surface_Base_Marker_Detector):

    def __getstate__(self):
        return (
            self.__apriltag_options,
            self.__marker_min_perimeter,
            self.__marker_detector_modes
        )

    def __setstate__(self, state):
        (
            self.__apriltag_options,
            self.__marker_min_perimeter,
            self.__marker_detector_modes
        ) = state
        self._detector = apriltag.Detector(detector_options=self.__apriltag_options)

    def __init__(
        self,
        marker_detector_modes: typing.Set[Surface_Marker_Detector_Mode],
        marker_min_perimeter: int = ...,
        apriltag_border: int = ...,
        apriltag_nthreads: int = ...,
        apriltag_quad_decimate: float = ...,
        apriltag_quad_blur: float = ...,
        apriltag_refine_edges: bool = ...,
        apriltag_refine_decode: bool = ...,
        apriltag_refine_pose: bool = ...,
        apriltag_debug: bool = ...,
        apriltag_quad_contours: bool = ...,
        **kwargs,
    ):
        apriltag_options = _Apriltag_V2_Detector_Options(
            families=DEFAULT_APRILTAG_FAMILY,
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
        state = (
            apriltag_options,
            marker_min_perimeter,
            marker_detector_modes,
        )
        self.__setstate__(state)

    @property
    def robust_detection(self) -> bool:
        return True

    @robust_detection.setter
    def robust_detection(self, value: bool):
        pass  # nop

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
    def marker_detector_modes(self) -> typing.Set[Surface_Marker_Detector_Mode]:
        return self.__marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: typing.Set[Surface_Marker_Detector_Mode]):
        self.__marker_detector_modes = value

    def detect_markers_iter(self, gray_img) -> typing.Iterable[Surface_Marker]:
        if Surface_Marker_Detector_Mode.APRILTAG_MARKER not in self.marker_detector_modes:
            return []
        markers = self._detector.detect(img=gray_img)
        markers = map(Surface_Marker.from_apriltag_v2_detection, markers)
        markers = filter(self._surface_marker_filter, markers)
        return markers


class Surface_Combined_Marker_Detector(Surface_Base_Marker_Detector):
    def __init__(
        self,
        marker_detector_modes: typing.Set[Surface_Marker_Detector_Mode],
        marker_min_perimeter: int = ...,
        square_marker_robust_detection: bool = ...,
        square_marker_inverted_markers: bool = ...,
        apriltag_families: str = ...,
        apriltag_border: int = ...,
        apriltag_nthreads: int = ...,
        apriltag_quad_decimate: float = ...,
        apriltag_quad_blur: float = ...,
        apriltag_refine_edges: bool = ...,
        apriltag_refine_decode: bool = ...,
        apriltag_refine_pose: bool = ...,
        apriltag_debug: bool = ...,
        apriltag_quad_contours: bool = ...,
    ):
        self.__square_detector = Surface_Square_Marker_Detector(
            marker_detector_modes=marker_detector_modes,
            marker_min_perimeter=marker_min_perimeter,
            square_marker_robust_detection=square_marker_robust_detection,
            square_marker_inverted_markers=square_marker_inverted_markers,
        )
        self.__apriltag_detector = Surface_Apriltag_V2_Marker_Detector(
            marker_detector_modes=marker_detector_modes,
            marker_min_perimeter=marker_min_perimeter,
            apriltag_families=apriltag_families,
            apriltag_border=apriltag_border,
            apriltag_nthreads=apriltag_nthreads,
            apriltag_quad_decimate=apriltag_quad_decimate,
            apriltag_quad_blur=apriltag_quad_blur,
            apriltag_refine_edges=apriltag_refine_edges,
            apriltag_refine_decode=apriltag_refine_decode,
            apriltag_refine_pose=apriltag_refine_pose,
            apriltag_debug=apriltag_debug,
            apriltag_quad_contours=apriltag_quad_contours,
        )

    @property
    def robust_detection(self) -> bool:
        return self.__square_detector.robust_detection

    @robust_detection.setter
    def robust_detection(self, value: bool):
        self.__square_detector.robust_detection = value

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
    def marker_detector_modes(self) -> typing.Set[Surface_Marker_Detector_Mode]:
        marker_detector_modes = self.__apriltag_detector.marker_detector_modes
        assert marker_detector_modes == self.__square_detector.marker_detector_modes
        return marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: typing.Set[Surface_Marker_Detector_Mode]):
        self.__square_detector.marker_detector_modes = value
        self.__apriltag_detector.marker_detector_modes = value

    def detect_markers_iter(self, gray_img) -> typing.Iterable[Surface_Marker]:
        return itertools.chain(
            self.__square_detector.detect_markers_iter(gray_img=gray_img),
            self.__apriltag_detector.detect_markers_iter(gray_img=gray_img),
        )


Surface_Marker_Detector = Surface_Combined_Marker_Detector
