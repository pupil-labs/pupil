import abc

import cv2
import numpy as np


class ImageManipulator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply_to(self, image, parameter):
        raise NotImplementedError


class ScaleTransform(ImageManipulator):
    def apply_to(self, image, parameter):
        """parameter: scale factor as float"""
        return cv2.resize(image, (0, 0), fx=parameter, fy=parameter)


class HorizontalFlip(ImageManipulator):
    def apply_to(self, image, parameter):
        """parameter: boolean indicating if image should be flipped"""
        return np.fliplr(image) if parameter else image


class VerticalFlip(ImageManipulator):
    def apply_to(self, image, parameter):
        """parameter: boolean indicating if image should be flipped"""
        return np.flipud(image) if parameter else image


class PupilRenderer(ImageManipulator):
    __slots__ = "pupil_getter"

    def __init__(self, pupil_getter):
        self.pupil_getter = pupil_getter

    def apply_to(self, image, parameter):
        """parameter: boolean indicating if pupil should be rendered"""
        if parameter:
            pupil_position = self.pupil_getter()
            if pupil_position:
                self.render_pupil(image, pupil_position)
        return image

    def render_pupil(self, image, pupil_position):
        el = pupil_position["ellipse"]
        conf = int(
            pupil_position.get(
                "model_confidence", pupil_position.get("confidence", 0.1)
            )
            * 255
        )
        el_points = self.get_ellipse_points((el["center"], el["axes"], el["angle"]))
        cv2.polylines(
            image,
            [np.asarray(el_points, dtype="i")],
            True,
            (0, 0, 255, conf),
            thickness=1,
        )
        cv2.circle(
            image,
            (int(el["center"][0]), int(el["center"][1])),
            5,
            (0, 0, 255, conf),
            thickness=-1,
        )

    @staticmethod
    def get_ellipse_points(e, num_pts=10):
        c1 = e[0][0]
        c2 = e[0][1]
        a = e[1][0]
        b = e[1][1]
        angle = e[2]

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
