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
import typing
import itertools

import square_marker_detect
import apriltag

from .surface_marker import Surface_Marker


__all__ = ["Surface_Marker_Detector"]


class Surface_Base_Marker_Detector(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def robust_detection(self) -> bool:
        # TODO: Remove external dependency on this property
        pass

    @robust_detection.setter
    @abc.abstractmethod
    def robust_detection(self, value: bool):
        # TODO: Remove external dependency on this property
        pass

    @property
    @abc.abstractmethod
    def inverted_markers(self) -> bool:
        # TODO: Remove external dependency on this property
        pass

    @inverted_markers.setter
    @abc.abstractmethod
    def inverted_markers(self, value: bool):
        # TODO: Remove external dependency on this property
        pass

    @abc.abstractmethod
    def detect_markers(self, gray_img) -> typing.List[Surface_Marker]:
        # TODO: Add type hints
        pass


class Surface_Square_Marker_Detector(Surface_Base_Marker_Detector):
    def __init__(
        self,
        square_marker_min_perimeter: int = ...,
        square_marker_robust_detection: bool = ...,
        square_marker_inverted_markers: bool = ...,
        **kwargs,
    ):
        self.__marker_min_perimeter = (
            square_marker_min_perimeter
            if square_marker_min_perimeter is not ...
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
        self.__previous_raw_markers = []

    @property
    def robust_detection(self) -> bool:
        # TODO: Remove external dependency on this property
        return self.__robust_detection

    @robust_detection.setter
    def robust_detection(self, value: bool):
        # TODO: Remove external dependency on this property
        self.__robust_detection = value

    @property
    def inverted_markers(self) -> bool:
        # TODO: Remove external dependency on this property
        return self.__inverted_markers

    @inverted_markers.setter
    def inverted_markers(self, value: bool):
        # TODO: Remove external dependency on this property
        self.__inverted_markers = value

    def detect_markers(self, gray_img) -> typing.Iterable[Surface_Marker]:
        # TODO: Add type hints

        grid_size = 5
        aperture = 11

        if self.__robust_detection:
            markers = square_marker_detect.detect_markers_robust(
                gray_img=gray_img,
                grid_size=grid_size,
                min_marker_perimeter=self.__marker_min_perimeter,
                aperture=aperture,
                prev_markers=self.__previous_raw_markers,
                true_detect_every_frame=3,
                invert_image=self.__inverted_markers,
            )
        else:
            markers = square_marker_detect.detect_markers(
                gray_img=gray_img,
                grid_size=grid_size,
                min_marker_perimeter=self.__marker_min_perimeter,
                aperture=aperture,
            )

        # Robust marker detection requires previous markers to be in a different
        # format than the surface tracker.
        self.__previous_raw_markers = markers
        return map(Surface_Marker.from_square_tag_detection, markers)


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
    def __init__(
        self,
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
        **kwargs,
    ):
        options = _Apriltag_V2_Detector_Options(
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
        self._detector = apriltag.Detector(detector_options=options)

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

    def detect_markers(self, gray_img) -> typing.Iterable[Surface_Marker]:
        markers = self._detector.detect(img=gray_img)
        return map(Surface_Marker.from_apriltag_v2_detection, markers)


class Surface_Combined_Marker_Detector(Surface_Base_Marker_Detector):
    def __init__(
        self,
        square_marker_min_perimeter: int = ...,
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
            square_marker_min_perimeter=square_marker_min_perimeter,
            square_marker_robust_detection=square_marker_robust_detection,
            square_marker_inverted_markers=square_marker_inverted_markers,
        )
        self.__apriltag_detector = Surface_Apriltag_V2_Marker_Detector(
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

    def detect_markers(self, gray_img) -> typing.Iterable[Surface_Marker]:
        return itertools.chain(
            self.__square_detector.detect_markers(gray_img=gray_img),
            self.__apriltag_detector.detect_markers(gray_img=gray_img),
        )


Surface_Marker_Detector = Surface_Combined_Marker_Detector
