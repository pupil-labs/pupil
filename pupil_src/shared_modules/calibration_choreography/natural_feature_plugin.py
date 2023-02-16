"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import os
import typing as T

import cv2
import glfw
import numpy as np
from gl_utils import GLFWErrorReporting, draw_circle_filled_func_builder

GLFWErrorReporting.set_default()

import audio
from methods import denormalize, normalize
from pyglui import ui
from pyglui.cygl.utils import RGBA

from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyAction,
    ChoreographyMode,
)
from .controller import GUIMonitor

logger = logging.getLogger(__name__)


class NaturalFeatureChoreographyPlugin(CalibrationChoreographyPlugin):
    """Calibrate using natural features in a scene.
    Features are selected by a user by clicking on
    """

    label = "Natural Feature Calibration"

    @classmethod
    def selection_label(cls) -> str:
        return "Natural Feature"

    @classmethod
    def selection_order(cls) -> float:
        return 3.0

    _NUMBER_OF_REF_POINTS_TO_CAPTURE = 30
    _RADIUS_OF_CIRCLE_DISPLAYED = 40.0

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool, **kwargs)
        self.__number_of_ref_points_gathered_from_last_click = 0
        self.__previously_detected_feature = None
        self.__feature_tracker = NaturalFeatureTracker()
        self._draw_circle_filled = draw_circle_filled_func_builder()

    def get_init_dict(self):
        return {}

    ### Public - Plugin

    @classmethod
    def _choreography_description_text(cls) -> str:
        return "Calibrate gaze parameters using features in your environment. Ask the subject to look at objects in the scene and click on them in the world window."

    def recent_events(self, events):
        super().recent_events(events)

        frame = events.get("frame")

        if not frame:
            return

        if not self.is_active:
            return

        # Always save pupil positions
        self.pupil_list.extend(events["pupil"])

        need_more_ref_points = (
            self.__number_of_ref_points_gathered_from_last_click
            < self._NUMBER_OF_REF_POINTS_TO_CAPTURE
        )

        detected_feature = self.__feature_tracker.update(frame.gray)

        if need_more_ref_points and detected_feature:
            ref = {}
            ref["screen_pos"] = detected_feature["img_pos"]
            ref["norm_pos"] = detected_feature["norm_pos"]
            ref["timestamp"] = frame.timestamp
            self.ref_list.append(ref)
            self.__previously_detected_feature = detected_feature
            self.__number_of_ref_points_gathered_from_last_click += 1
        else:
            self.__previously_detected_feature = None

        # Update UI
        self.status_text = (
            "Sampling Gaze Data"
            if len(self.ref_list)
            else "Click to Sample at Location"
        )

    def gl_display(self):
        feature = self.__previously_detected_feature
        if feature:
            recent_frame_size = self.g_pool.capture.frame_size
            point = denormalize(feature["norm_pos"], recent_frame_size, flip_y=True)
            self._draw_circle_filled(
                tuple(point),
                size=self._RADIUS_OF_CIRCLE_DISPLAYED / 2,
                color=RGBA(0.0, 1.0, 0.0, 0.5),
            )

    def on_click(self, pos, button, action):
        if action == glfw.PRESS and self.is_active:
            self.__feature_tracker.reset(pos)
            self.__number_of_ref_points_gathered_from_last_click = 0
            return True  # click consumed
        return False  # click not consumed

    def _perform_start(self):
        if not self.g_pool.capture.online:
            logger.error(
                f"{self.current_mode.label} requiers world capture video input."
            )
            return
        return super()._perform_start()


class NaturalFeatureTracker:
    def __init__(self):
        self.reset(detection_point=None)

    def reset(self, detection_point: T.Optional[T.Tuple[float, float]]):
        points = (
            np.array([detection_point], dtype=np.float32) if detection_point else None
        )
        self.__previous_gray_image = None
        self.__previous_detected_points = points
        self.__was_detected_on_last_update = False

    def update(self, gray_img) -> T.Optional[dict]:
        # No starting point was selected yet
        if self.__previous_detected_points is None:
            return None

        if self.__previous_gray_image is None:
            self.__previous_gray_image = gray_img.copy()

        self.__was_detected_on_last_update = False

        # in cv2.3 nextPts is falsly required as an argument.
        next_points_dummy = self.__previous_detected_points.copy()
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.__previous_gray_image,
            gray_img,
            self.__previous_detected_points,
            next_points_dummy,
            winSize=(100, 100),
        )

        if status[0]:
            self.__was_detected_on_last_update = True
            self.__previous_gray_image = gray_img.copy()
            self.__previous_detected_points = next_points

            screen_point = next_points[0].tolist()  # we prefer python types.

            normal_point = normalize(
                screen_point, (gray_img.shape[1], gray_img.shape[0]), flip_y=True
            )

            return {"img_pos": screen_point, "norm_pos": normal_point}

        else:
            return None
