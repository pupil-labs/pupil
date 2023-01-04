"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import enum
import logging
import typing as T
from platform import system

import audio
import cv2
import glfw
import numpy as np
import OpenGL.GL as gl
from circle_detector import CircleTracker
from gl_utils import GLFWErrorReporting, adjust_gl_view, basic_gl_setup, clear_gl_screen
from plugin import Plugin
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_polyline
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

GLFWErrorReporting.set_default()

from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyAction,
    ChoreographyMode,
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


class SingleMarkerMode(enum.Enum):
    FULL_SCREEN = "Full screen"
    WINDOW = "Window"
    MANUAL = "Physical Marker"

    @staticmethod
    def all_modes() -> T.List["SingleMarkerMode"]:
        return sorted(SingleMarkerMode, key=lambda m: m.order)

    @staticmethod
    def from_label(label: str) -> "SingleMarkerMode":
        return SingleMarkerMode(label)

    @property
    def label(self) -> str:
        return self.value

    @property
    def order(self) -> float:
        if self == SingleMarkerMode.MANUAL:
            return 1.0
        elif self == SingleMarkerMode.FULL_SCREEN:
            return 2.0
        elif self == SingleMarkerMode.WINDOW:
            return 3.0
        else:
            return float("inf")


class SingleMarkerChoreographyPlugin(
    MonitorSelectionMixin, CalibrationChoreographyPlugin
):
    """Calibrate using a single marker.
    Move your head for example in a spiral motion while gazing
    at the marker to quickly sample a wide range gaze angles.
    """

    label = "Single Marker Calibration"

    @classmethod
    def selection_label(cls) -> str:
        return "Single Marker"

    @classmethod
    def selection_order(cls) -> float:
        return 2.0

    _STOP_MARKER_FRAMES_NEEDED_TO_STOP = 30
    _FIXED_MARKER_POSITION = (0.5, 0.5)

    def __init__(
        self,
        g_pool,
        marker_mode=None,
        marker_scale=1.0,
        sample_duration=40,
        monitor_name=None,
        **kwargs,
    ):
        if marker_mode is None:
            marker_mode = SingleMarkerMode.MANUAL
        else:
            marker_mode = SingleMarkerMode.from_label(marker_mode)

        super().__init__(g_pool, **kwargs)

        # Public properties
        self.selected_monitor_name = monitor_name
        self.marker_mode = marker_mode

        # Private properties
        self.__previously_detected_markers = []
        self.__circle_tracker = CircleTracker()
        self.__auto_stop_tracker = AutoStopTracker(
            markers_needed=self._STOP_MARKER_FRAMES_NEEDED_TO_STOP
        )
        self.__marker_window = MarkerWindowController(marker_scale=marker_scale)
        self.__marker_window.add_observer(
            "on_window_did_close", self._on_window_did_close
        )

    def get_init_dict(self):
        d = {}
        d["marker_mode"] = self.marker_mode.label
        d["marker_scale"] = self.__marker_window.marker_scale
        d["monitor_name"] = self.selected_monitor_name
        return d

    def cleanup(self):
        super().cleanup()

    @property
    def marker_mode(self) -> SingleMarkerMode:
        return self.__marker_mode

    @marker_mode.setter
    def marker_mode(self, value: SingleMarkerMode):
        self.__marker_mode = value
        self._ui_update_visibility_digital_marker_config()

    ### Public - Plugin

    @classmethod
    def _choreography_description_text(cls) -> str:
        return "Calibrate using a single marker. Gaze at the center of the marker and move your head (e.g. in a slow spiral movement). This calibration method enables you to quickly sample a wide range of gaze angles and cover a large range of your FOV."

    def _init_custom_menu_ui_elements(self) -> list:

        self.__ui_selector_marker_mode = ui.Selector(
            "marker_mode",
            self,
            label="Marker display mode",
            labels=[m.label for m in SingleMarkerMode.all_modes()],
            selection=SingleMarkerMode.all_modes(),
        )

        # TODO: potential race condition through selection_getter. Should ensure
        # that current selection will always be present in the list returned by the
        # selection_getter. Highly unlikely though as this needs to happen between
        # having clicked the Selector and the next redraw.
        # See https://github.com/pupil-labs/pyglui/pull/112/commits/587818e9556f14bfedd8ff8d093107358745c29b
        self.__ui_selector_monitor_name = ui.Selector(
            "selected_monitor_name",
            self,
            label="Monitor",
            labels=self.currently_connected_monitor_names(),
            selection=self.currently_connected_monitor_names(),
        )

        self.__ui_slider_marker_scale = ui.Slider(
            "marker_scale",
            self.__marker_window,
            label="Marker size",
            min=0.5,
            max=2.0,
            step=0.1,
        )

        return [
            self.__ui_selector_marker_mode,
            self.__ui_selector_monitor_name,
            self.__ui_slider_marker_scale,
        ]

    def init_ui(self):
        super().init_ui()
        # Save UI elements that are part of the digital marker config
        self.__ui_digital_marker_config_elements = [
            self.__ui_selector_monitor_name,
            self.__ui_slider_marker_scale,
        ]
        # Save start index of the UI elements of digital marker config
        self.__ui_digital_marker_config_start_index = min(
            self.menu.elements.index(elem)
            for elem in self.__ui_digital_marker_config_elements
        )
        self._ui_update_visibility_digital_marker_config()

    def _ui_update_visibility_digital_marker_config(self):
        try:
            ui_menu = self.menu
            ui_elements = self.__ui_digital_marker_config_elements
            start_index = self.__ui_digital_marker_config_start_index
        except AttributeError:
            return

        is_visible = self.marker_mode != SingleMarkerMode.MANUAL

        for i, ui_element in enumerate(ui_elements):
            index = start_index + i
            if is_visible and ui_element not in ui_menu:
                ui_menu.insert(index, ui_element)
                continue
            if not is_visible and ui_element in ui_menu:
                ui_menu.remove(ui_element)
                continue

    def deinit_ui(self):
        self.__marker_window.close_window()
        super().deinit_ui()

    def recent_events(self, events):
        super().recent_events(events)

        frame = events.get("frame")
        should_animate = True
        state = self.__marker_window.window_state

        if not frame:
            return

        self.__marker_window.update_state()

        if not self.is_active:
            # If the plugin is not active, just return
            return

        if self.marker_mode == SingleMarkerMode.MANUAL:
            assert isinstance(
                state, MarkerWindowStateClosed
            ), "In manual mode, window should be closed at all times."

        if isinstance(state, MarkerWindowStateClosed):
            if self.marker_mode != SingleMarkerMode.MANUAL:
                # This state should be unreachable, since there is an early return if the plugin is inactive
                assert not self.is_active
                return

        elif isinstance(state, MarkerWindowStateOpened):
            assert self.is_active  # Sanity check
            assert self.marker_mode != SingleMarkerMode.MANUAL
            pass  # Continue with processing the frame

        else:
            raise UnhandledMarkerWindowStateError(state)

        # Always save pupil positions
        self.pupil_list.extend(events["pupil"])

        gray_img = frame.gray

        # Update the marker
        ref_marker, stop_marker = self.__detect_ref_marker_and_stop_marker(gray_img)
        self.__auto_stop_tracker.process_markers(stop_marker)

        # Stop if autostop condition is satisfied
        if self.__auto_stop_tracker.should_stop:
            self._signal_should_stop(mode=self.current_mode)

        # Signal marker window controller that a marker was detected (for feedback)
        self.__marker_window.is_marker_detected = ref_marker is not None

        should_save_ref_marker = False

        if isinstance(state, MarkerWindowStateClosed):
            should_save_ref_marker = self.marker_mode == SingleMarkerMode.MANUAL

        elif isinstance(state, MarkerWindowStateIdle):
            self.__marker_window.show_marker(
                self._FIXED_MARKER_POSITION, should_animate=should_animate
            )

        elif isinstance(state, MarkerWindowStateAnimatingInMarker):
            pass  # No-op

        elif isinstance(state, MarkerWindowStateShowingMarker):
            assert self.marker_mode != SingleMarkerMode.MANUAL
            should_save_ref_marker = True

        elif isinstance(state, MarkerWindowStateAnimatingOutMarker):
            pass  # No-op

        else:
            raise UnhandledMarkerWindowStateError(state)

        if should_save_ref_marker and ref_marker is not None and stop_marker is None:
            ref = {}
            ref["norm_pos"] = ref_marker["norm_pos"]
            ref["screen_pos"] = ref_marker["img_pos"]
            ref["timestamp"] = frame.timestamp
            self.ref_list.append(ref)

        # Update UI
        self.__marker_window.draw_window()
        self.status_text = self._FIXED_MARKER_POSITION if self.is_active else None

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if not self.is_active:
            return

        for marker in self.__previously_detected_markers:
            # draw the largest ellipse of all detected markers
            e = marker["ellipses"][-1]
            pts = cv2.ellipse2Poly(
                (int(e[0][0]), int(e[0][1])),
                (int(e[1][0] / 2), int(e[1][1] / 2)),
                int(e[-1]),
                0,
                360,
                15,
            )
            draw_polyline(pts, color=RGBA(0.0, 1.0, 0, 1.0))
            if len(self.__previously_detected_markers) > 1:
                draw_polyline(pts, 1, RGBA(1.0, 0.0, 0.0, 0.5), line_type=gl.GL_POLYGON)

            # draw indicator on the stop marker(s)
            if marker["marker_type"] == "Stop":
                e = marker["ellipses"][-1]
                pts = cv2.ellipse2Poly(
                    (int(e[0][0]), int(e[0][1])),
                    (int(e[1][0] / 2), int(e[1][1] / 2)),
                    int(e[-1]),
                    0,
                    360,
                    360 // self._STOP_MARKER_FRAMES_NEEDED_TO_STOP,
                )
                indicator = (
                    [e[0]]
                    + pts[self.__auto_stop_tracker.detected_count :].tolist()
                    + [e[0]]
                )
                draw_polyline(
                    indicator, color=RGBA(8.0, 0.1, 0.1, 0.8), line_type=gl.GL_POLYGON
                )

    ### Internal

    def _perform_start(self):
        if not self.g_pool.capture.online:
            logger.error(
                f"{self.current_mode.label} requiers world capture video input."
            )
            return

        self.__auto_stop_tracker.reset()

        super()._perform_start()

        if self.marker_mode != SingleMarkerMode.MANUAL:
            is_fullscreen = self.marker_mode == SingleMarkerMode.FULL_SCREEN
            self.__marker_window.open_window(
                title=self.current_mode.label,
                monitor_name=self.selected_monitor_name,
                is_fullscreen=is_fullscreen,
            )

    def _perform_stop(self):
        self.__marker_window.close_window()
        super()._perform_stop()

    ### Private

    def _on_window_did_close(self):
        self._signal_should_stop(mode=self.current_mode)

    def __detect_ref_marker_and_stop_marker(
        self, gray_img
    ) -> T.Tuple[T.Optional[dict], T.Optional[dict]]:
        markers = self.__circle_tracker.update(gray_img)
        ref_marker = None
        stop_marker = None

        # Check if there are more than one markers
        if len(markers) > 1:
            logger.warning(
                f"{len(markers)} markers detected. Please remove all the other markers"
            )

        for marker in markers:
            if marker["marker_type"] == "Ref":
                ref_marker = marker
            if marker["marker_type"] == "Stop":
                stop_marker = marker

        self.__previously_detected_markers = [
            m for m in [ref_marker, stop_marker] if m is not None
        ]

        return ref_marker, stop_marker


class AutoStopTracker:
    def __init__(self, markers_needed: int):
        self.__stop_marker_count = 0
        self.__stop_markers_needed_to_stop = markers_needed

    @property
    def should_stop(self) -> bool:
        return self.__stop_marker_count >= self.__stop_markers_needed_to_stop

    @property
    def detected_count(self) -> int:
        return self.__stop_marker_count

    def reset(self):
        self.__stop_marker_count = 0

    def process_markers(self, markers):
        if not markers:
            markers = []
        elif isinstance(markers, dict):
            markers = [markers]

        markers = [m for m in markers if m and m["marker_type"] == "Stop"]
        markers_count = len(markers)

        if markers_count > 0:
            self.__stop_marker_count += markers_count
        else:
            self.reset()
