#!/usr/bin/env python

"""
Modified version of https://github.com/swatbotics/apriltag/blob/427a42ce24578e0fcc483eaf35794b986352070d/python/apriltag.py

Python wrapper for C version of apriltags. This program creates two
classes that are used to detect apriltags and extract information from
them. Using this module, you can identify all apriltags visible in an
image, and get information about the location and orientation of the
tags.

Original author: Isaac Dulin, Spring 2016
Updates:
- Matt Zucker, Fall 2016
- Pupil Labs, Spring 2019
"""

import collections
import ctypes
import os
import re

import numpy as np

DetectionBase = collections.namedtuple(
    "DetectionBase",
    [
        "tag_family",
        "tag_id",
        "hamming",
        "goodness",
        "decision_margin",
        "homography",
        "center",
        "corners",
    ],
)


class DetectorOptions:
    """Convience wrapper for object to pass into Detector
    initializer. You can also pass in the output of an
    argparse.ArgumentParser on which you have called add_arguments.
    """

    def __init__(
        self,
        families="tag36h11",
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=False,
        refine_pose=False,
        debug=False,
        quad_contours=True,
    ):
        self.families = families
        self.border = int(border)

        self.nthreads = int(nthreads)
        self.quad_decimate = float(quad_decimate)
        self.quad_sigma = float(quad_blur)
        self.refine_edges = int(refine_edges)
        self.refine_decode = int(refine_decode)
        self.refine_pose = int(refine_pose)
        self.debug = int(debug)
        self.quad_contours = quad_contours


class Detector:
    """Pythonic wrapper for apriltag_detector. Initialize by passing in
    the output of an argparse.ArgumentParser on which you have called
    add_arguments; or an instance of the DetectorOptions class.  You can
    also optionally pass in a list of paths to search for the C dynamic
    library used by ctypes.
    """

    def __init__(self, detector_options=DetectorOptions()):
        self.tag_detector = None
        filename = ctypes.util.find_library("apriltag")
        try:
            self.libc = ctypes.CDLL(filename)
        except (OSError, TypeError) as err:
            raise RuntimeError("apriltag dependencies not found") from err
        if self.libc is None:
            raise RuntimeError("could not find DLL named " + filename)

        self._declare_libc_function()

        self._create_apriltag_detector_object(detector_options)
        if detector_options.quad_contours:
            self.libc.apriltag_detector_enable_quad_contours(self.tag_detector, 1)

        if isinstance(detector_options.families, list):
            families_list = detector_options.families
        else:
            families_list = [
                n for n in re.split(r"\W+", detector_options.families) if n
            ]

        for family in families_list:
            self.add_tag_family(detector_options, family)

    def _declare_libc_function(self):
        self.libc.apriltag_detector_create.restype = ctypes.POINTER(_ApriltagDetector)
        self.libc.apriltag_family_create.restype = ctypes.POINTER(_ApriltagFamily)
        self.libc.apriltag_detector_detect.restype = ctypes.POINTER(_ZArray)
        self.libc.image_u8_create.restype = ctypes.POINTER(_ImageU8)
        self.libc.image_u8_write_pnm.restype = ctypes.c_int
        self.libc.apriltag_family_list.restype = ctypes.POINTER(_ZArray)
        self.libc.apriltag_vis_detections.restype = None

        self.libc.pose_from_homography.restype = ctypes.POINTER(_Matd)
        self.libc.matd_create.restype = ctypes.POINTER(_Matd)

    def _create_apriltag_detector_object(self, detector_options):
        self.tag_detector = self.libc.apriltag_detector_create()
        self.tag_detector.contents.nthreads = int(detector_options.nthreads)
        self.tag_detector.contents.quad_decimate = float(detector_options.quad_decimate)
        self.tag_detector.contents.quad_sigma = float(detector_options.quad_sigma)
        self.tag_detector.refine_edges = int(detector_options.refine_edges)
        self.tag_detector.refine_decode = int(detector_options.refine_decode)
        self.tag_detector.refine_pose = int(detector_options.refine_pose)

    def add_tag_family(self, detector_options, name):
        """Add a single tag family to this detector."""

        family = self.libc.apriltag_family_create(name.encode("ascii"))

        if family:
            family.contents.border = detector_options.border
            self.libc.apriltag_detector_add_family(self.tag_detector, family)
        else:
            print("Unrecognized tag family name. Try e.g. tag36h11")

    def __del__(self):
        if self.tag_detector is not None:
            self.libc.apriltag_detector_destroy(self.tag_detector)

    def detect(self, img):
        """Run detectons on the provided image. The image must be a grayscale
        image of type np.uint8.
        """

        assert len(img.shape) == 2
        assert img.dtype == np.uint8

        c_img = self._convert_image(img)

        apriltag_detections = self.libc.apriltag_detector_detect(
            self.tag_detector, c_img
        )
        apriltag = ctypes.POINTER(_ApriltagDetection)()
        detections = []
        for i in range(apriltag_detections.contents.size):
            self.libc.zarray_get(apriltag_detections, i, ctypes.byref(apriltag))
            tag = apriltag.contents
            homography = _matd_get_array(tag.H).copy()
            center = np.ctypeslib.as_array(tag.c, shape=(2,)).copy()
            corners = np.ctypeslib.as_array(tag.p, shape=(4, 2)).copy()

            detection = DetectionBase(
                ctypes.string_at(tag.family.contents.name),
                tag.id,
                tag.hamming,
                tag.goodness,
                tag.decision_margin,
                homography,
                center,
                corners,
            )
            detections.append(detection)

        self.libc.image_u8_destroy(c_img)
        self.libc.apriltag_detections_destroy(apriltag_detections)
        return detections

    def _convert_image(self, img):
        height, width = img.shape
        c_img = self.libc.image_u8_create(width, height)
        tmp = _image_u8_get_array(c_img)

        # copy the opencv image into the destination array, accounting for the
        # difference between stride & width.
        tmp[:, :width] = img

        # tmp goes out of scope here but we don't care because
        # the underlying data is still in c_img.
        return c_img


class _ImageU8(ctypes.Structure):
    """Wraps image_u8 C struct."""

    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("buf", ctypes.POINTER(ctypes.c_uint8)),
    ]


class _Matd(ctypes.Structure):
    """Wraps matd C struct."""

    _fields_ = [
        ("nrows", ctypes.c_int),
        ("ncols", ctypes.c_int),
        ("data", ctypes.c_double * 1),
    ]


class _ZArray(ctypes.Structure):
    """Wraps zarray C struct."""

    _fields_ = [
        ("el_sz", ctypes.c_size_t),
        ("size", ctypes.c_int),
        ("alloc", ctypes.c_int),
        ("data", ctypes.c_void_p),
    ]


class _ApriltagFamily(ctypes.Structure):
    """Wraps apriltag_family C struct."""

    _fields_ = [
        ("ncodes", ctypes.c_int32),
        ("codes", ctypes.POINTER(ctypes.c_int64)),
        ("black_border", ctypes.c_int32),
        ("d", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("name", ctypes.c_char_p),
    ]


class _ApriltagDetection(ctypes.Structure):
    """Wraps apriltag_detection C struct."""

    _fields_ = [
        ("family", ctypes.POINTER(_ApriltagFamily)),
        ("id", ctypes.c_int),
        ("hamming", ctypes.c_int),
        ("goodness", ctypes.c_float),
        ("decision_margin", ctypes.c_float),
        ("H", ctypes.POINTER(_Matd)),
        ("c", ctypes.c_double * 2),
        ("p", (ctypes.c_double * 2) * 4),
    ]


class _ApriltagDetector(ctypes.Structure):
    """Wraps apriltag_detector C struct."""

    _fields_ = [
        ("nthreads", ctypes.c_int),
        ("quad_decimate", ctypes.c_float),
        ("quad_sigma", ctypes.c_float),
        ("refine_edges", ctypes.c_int),
        ("refine_decode", ctypes.c_int),
        ("refine_pose", ctypes.c_int),
        ("debug", ctypes.c_int),
        ("quad_contours", ctypes.c_int),
    ]


def _ptr_to_array2d(datatype, ptr, rows, cols):
    array_type = (datatype * cols) * rows
    array_buf = array_type.from_address(ctypes.addressof(ptr))
    return np.ctypeslib.as_array(array_buf, shape=(rows, cols))


def _image_u8_get_array(img_ptr):
    return _ptr_to_array2d(
        ctypes.c_uint8,
        img_ptr.contents.buf.contents,
        img_ptr.contents.height,
        img_ptr.contents.stride,
    )


def _matd_get_array(mat_ptr):
    return _ptr_to_array2d(
        ctypes.c_double,
        mat_ptr.contents.data,
        int(mat_ptr.contents.nrows),
        int(mat_ptr.contents.ncols),
    )
