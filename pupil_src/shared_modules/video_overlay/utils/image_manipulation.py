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
import logging

import cv2
import numpy as np
from pupil_detector_plugins import color_scheme

logger = logging.getLogger(__name__)


def float_to_int(value: float) -> int:
    return int(value) if np.isfinite(value) else 0


class ImageManipulator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply_to(self, image, parameter, **kwargs):
        raise NotImplementedError


class ScaleTransform(ImageManipulator):
    def apply_to(self, image, parameter, **kwargs):
        """parameter: scale factor as float"""
        return cv2.resize(image, (0, 0), fx=parameter, fy=parameter)


class HorizontalFlip(ImageManipulator):
    def apply_to(self, image, parameter, *, is_fake_frame, **kwargs):
        """parameter: boolean indicating if image should be flipped"""
        if parameter and not is_fake_frame:
            return np.fliplr(image)
        else:
            return image


class VerticalFlip(ImageManipulator):
    def apply_to(self, image, parameter, *, is_fake_frame, **kwargs):
        """parameter: boolean indicating if image should be flipped"""
        if parameter and not is_fake_frame:
            return np.flipud(image)
        else:
            return image


class PupilRenderer(ImageManipulator):
    __slots__ = "pupil_getter"

    def __init__(self, pupil_getter):
        self.pupil_getter = pupil_getter

    def apply_to(self, image, parameter, *, is_fake_frame, **kwargs):
        """parameter: boolean indicating if pupil should be rendered"""
        if parameter and not is_fake_frame:
            pupil_pos_2d, pupil_pos_3d = self.pupil_getter()
            if pupil_pos_2d:
                self.render_pupil_2d(image, pupil_pos_2d)
            if pupil_pos_3d:
                self.render_pupil_3d(image, pupil_pos_3d)
        return image

    def render_pupil_2d(self, image, pupil_position):
        el = pupil_position["ellipse"]

        conf = pupil_position["confidence"] * 255
        conf = float_to_int(conf)

        if conf > 0:
            self.render_ellipse(
                image, el, color=color_scheme.PUPIL_ELLIPSE_2D.flip_c0_c2
            )

    def render_pupil_3d(self, image, pupil_position):
        el = pupil_position["ellipse"]

        conf = pupil_position["confidence"] * 255
        conf = float_to_int(conf)

        if conf > 0:
            self.render_ellipse(
                image, el, color=color_scheme.PUPIL_ELLIPSE_3D.flip_c0_c2
            )

        if pupil_position["model_confidence"] <= 0.0:
            # NOTE: if 'model_confidence' == 0, some values of the 'projected_sphere'
            # might be 'nan', which will cause cv2.ellipse to crash.
            # TODO: Fix in detectors.
            return

        model_confidence_threshold = 0.6
        color = color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_IN.flip_c0_c2
        if pupil_position["model_confidence"] < model_confidence_threshold:
            color = color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_OUT.flip_c0_c2

        eye_ball = pupil_position.get("projected_sphere", None)
        if eye_ball is not None:
            try:
                cv2.ellipse(
                    image,
                    center=tuple(int(v) for v in eye_ball["center"]),
                    axes=tuple(int(v / 2) for v in eye_ball["axes"]),
                    angle=int(eye_ball["angle"]),
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=2,
                )
            except Exception as e:
                # Known issues:
                #   - There are reports of negative eye_ball axes, raising cv2.error.
                #     TODO: Investigate cause in detectors.
                logger.debug(
                    "Error rendering 3D eye-ball outline! Skipping...\n"
                    f"eye_ball: {eye_ball}\n"
                    f"{type(e)}: {e}"
                )

    def render_ellipse(self, image, ellipse, color):
        if not all(np.isfinite(ellipse["center"])):
            return
        if not all(np.isfinite(ellipse["axes"])):
            return
        if not np.isfinite(ellipse["angle"]):
            return

        outline = self.get_ellipse_points(
            ellipse["center"], ellipse["axes"], ellipse["angle"]
        )
        outline = [np.asarray(outline, dtype="i")]
        cv2.polylines(image, outline, True, color, thickness=1)

        center = (int(ellipse["center"][0]), int(ellipse["center"][1]))
        cv2.circle(image, center, 5, color, thickness=-1)

    @staticmethod
    def get_ellipse_points(center, axes, angle, num_pts=10):
        c1 = center[0]
        c2 = center[1]
        a = axes[0]
        b = axes[1]

        steps = np.linspace(0, 2 * np.pi, num=num_pts, endpoint=False)
        rot = cv2.getRotationMatrix2D((0, 0), -angle, 1)

        pts1 = a / 2.0 * np.cos(steps)
        pts2 = b / 2.0 * np.sin(steps)
        pts = np.column_stack((pts1, pts2, np.ones(pts1.shape[0])))

        pts_rot = np.matmul(rot, pts.T)
        pts_rot = pts_rot.T

        pts_rot[:, 0] += c1
        pts_rot[:, 1] += c2

        return pts_rot
