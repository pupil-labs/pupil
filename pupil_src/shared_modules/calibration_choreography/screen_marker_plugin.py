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
import typing as T

import cv2
import glfw
import numpy as np
import OpenGL.GL as gl
from gl_utils import GLFWErrorReporting, adjust_gl_view, basic_gl_setup, clear_gl_screen

GLFWErrorReporting.set_default()

from platform import system

import audio
from circle_detector import CircleTracker
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_polyline
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyAction,
    ChoreographyMode,
    ChoreographyNotification,
)
from .controller import (
    GUIMonitor,
    MarkerWindowController,
    MarkerWindowStateAnimatingInMarker,
    MarkerWindowStateAnimatingOutMarker,
    MarkerWindowStateClosed,
    MarkerWindowStateIdle,
    MarkerWindowStateOpened,
    MarkerWindowStateShowingMarker,
    UnhandledMarkerWindowStateError,
)
from .mixin import MonitorSelectionMixin

logger = logging.getLogger(__name__)


class ScreenMarkerChoreographyPlugin(
    MonitorSelectionMixin, CalibrationChoreographyPlugin
):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between
    """

    label = "Screen Marker Calibration"

    @classmethod
    def selection_label(cls) -> str:
        return "Screen Marker"

    @classmethod
    def selection_order(cls) -> float:
        return 1.0

    @staticmethod
    def get_list_of_markers_to_show(mode: ChoreographyMode) -> list:
        if ChoreographyMode.CALIBRATION == mode:
            return [(0.5, 0.5), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        if ChoreographyMode.VALIDATION == mode:
            return [(0.5, 1.0), (1.0, 0.5), (0.5, 0.0), (0.0, 0.5)]
        raise ValueError(f"Unknown mode {mode}")

    def __init__(
        self,
        g_pool,
        fullscreen=True,
        marker_scale=1.0,
        sample_duration=40,
        monitor_name=None,
        **kwargs,
    ):
        super().__init__(g_pool, **kwargs)

        # Public properties
        self.selected_monitor_name = monitor_name
        self.is_fullscreen = fullscreen
        self.sample_duration = sample_duration

        # Private properties
        self.__current_list_of_markers_to_show = []
        self.__currently_shown_marker_position = None
        self.__ref_count_for_current_marker_position = 0

        self.__previously_detected_markers = []
        self.__circle_tracker = CircleTracker()
        self.__marker_window = MarkerWindowController(marker_scale=marker_scale)
        self.__marker_window.add_observer(
            "on_window_did_close", self._on_window_did_close
        )

    def get_init_dict(self):
        d = {}
        d["fullscreen"] = self.is_fullscreen
        d["marker_scale"] = self.__marker_window.marker_scale
        d["monitor_name"] = self.selected_monitor_name
        return d

    ### Public - Plugin

    @classmethod
    def _choreography_description_text(cls) -> str:
        return "Calibrate gaze parameters using a screen based animation."

    def _init_custom_menu_ui_elements(self) -> list:

        self.__ui_selector_monitor_name = ui.Selector(
            "selected_monitor_name",
            self,
            label="Monitor",
            labels=self.currently_connected_monitor_names(),
            selection=self.currently_connected_monitor_names(),
        )

        self.__ui_switch_is_fullscreen = ui.Switch(
            "is_fullscreen", self, label="Use fullscreen"
        )

        self.__ui_slider_marker_scale = ui.Slider(
            "marker_scale",
            self.__marker_window,
            label="Marker size",
            min=0.5,
            max=2.0,
            step=0.1,
        )

        self.__ui_slider_sample_duration = ui.Slider(
            "sample_duration", self, label="Sample duration", min=10, max=100, step=1
        )

        return [
            self.__ui_selector_monitor_name,
            self.__ui_switch_is_fullscreen,
            self.__ui_slider_marker_scale,
            self.__ui_slider_sample_duration,
        ]

    def deinit_ui(self):
        self.__marker_window.close_window()
        super().deinit_ui()

    def recent_events(self, events):
        super().recent_events(events)

        frame = events.get("frame")
        state = self.__marker_window.window_state
        should_animate = True

        if not frame:
            return

        self.__marker_window.update_state()

        if isinstance(state, MarkerWindowStateClosed):
            return

        elif isinstance(state, MarkerWindowStateOpened):
            assert self.is_active  # Sanity check
            pass  # Continue with processing the frame

        else:
            raise UnhandledMarkerWindowStateError(state)

        # Always save pupil positions
        self.pupil_list.extend(events["pupil"])

        # Detect reference circle marker
        detected_marker = self.__detect_reference_circle_marker(frame.gray)

        # Signal marker window controller that a marker was detected (for feedback)
        self.__marker_window.is_marker_detected = detected_marker is not None

        if isinstance(state, MarkerWindowStateIdle):
            assert self.__currently_shown_marker_position is None  # Sanity check
            if self.__current_list_of_markers_to_show:
                self.__currently_shown_marker_position = (
                    self.__current_list_of_markers_to_show.pop(0)
                )
                logger.debug(
                    f"Moving screen marker to site at {self.__currently_shown_marker_position}"
                )
                self.__marker_window.show_marker(
                    marker_position=self.__currently_shown_marker_position,
                    should_animate=should_animate,
                )
                return
            else:
                # No more markers to show; stop calibration choreography.
                self._signal_should_stop(mode=self.current_mode)
                return

        if isinstance(state, MarkerWindowStateAnimatingInMarker):
            assert self.__currently_shown_marker_position is not None  # Sanity check
            pass  # No-op

        elif isinstance(state, MarkerWindowStateShowingMarker):
            assert self.__currently_shown_marker_position is not None  # Sanity check

            if detected_marker is not None:
                ref = {}
                ref["norm_pos"] = detected_marker["norm_pos"]
                ref["screen_pos"] = detected_marker["img_pos"]
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)

            should_move_to_next_marker = len(self.ref_list) == self.sample_duration * (
                self.__ref_count_for_current_marker_position + 1
            )

            if should_move_to_next_marker:
                # Finished collecting samples for current active site
                self.__currently_shown_marker_position = None
                self.__ref_count_for_current_marker_position += 1
                self.__marker_window.hide_marker(should_animate=should_animate)

        elif isinstance(state, MarkerWindowStateAnimatingOutMarker):
            assert self.__currently_shown_marker_position is None  # Sanity check
            pass  # No-op

        else:
            raise UnhandledMarkerWindowStateError(state)

        # Update UI
        self.__marker_window.draw_window()
        self.status_text = self.__currently_shown_marker_position

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        # debug mode within world will show green ellipses around detected ellipses
        if not self.is_active:
            return

        markers = self.__previously_detected_markers

        for marker in markers:
            e = marker["ellipses"][-1]  # outermost ellipse
            pts = cv2.ellipse2Poly(
                (int(e[0][0]), int(e[0][1])),
                (int(e[1][0] / 2), int(e[1][1] / 2)),
                int(e[-1]),
                0,
                360,
                15,
            )
            draw_polyline(pts, 1, RGBA(0.0, 1.0, 0.0, 1.0))
            if len(markers) > 1:
                draw_polyline(pts, 1, RGBA(1.0, 0.0, 0.0, 0.5), line_type=gl.GL_POLYGON)

    ### Internal

    def _perform_start(self):
        if not self.g_pool.capture.online:
            logger.error(
                f"{self.current_mode.label} requiers world capture video input."
            )
            return

        self.__current_list_of_markers_to_show = self.get_list_of_markers_to_show(
            mode=self.current_mode,
        )
        self.__currently_shown_marker_position = None
        self.__ref_count_for_current_marker_position = 0

        super()._perform_start()

        self.__marker_window.open_window(
            title=self.current_mode.label,
            monitor_name=self.selected_monitor_name,
            is_fullscreen=self.is_fullscreen,
        )

    def _perform_stop(self):
        self.__marker_window.close_window()
        super()._perform_stop()

    ### Private

    def _on_window_did_close(self):
        self._signal_should_stop(mode=self.current_mode)

    def __detect_reference_circle_marker(self, gray_img):

        # Detect all circular markers
        circle_markers = self.__circle_tracker.update(gray_img)

        # Only keep Ref markers
        circle_markers = [
            marker for marker in circle_markers if marker["marker_type"] == "Ref"
        ]

        # Store detected Ref markers for debugging/visualization
        self.__previously_detected_markers = circle_markers

        if len(circle_markers) == 0:
            return None
        elif len(circle_markers) == 1:
            return circle_markers[0]
        else:
            logger.warning(
                f"{len(circle_markers)} markers detected. Please remove all the other markers"
            )
            return circle_markers[0]
