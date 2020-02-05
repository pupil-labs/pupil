"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

import cv2
import numpy as np
from gl_utils import adjust_gl_view, clear_gl_screen, basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from circle_detector import CircleTracker
from platform import system

import audio

from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_polyline, RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

from .controller import GUIMonitor
from .controller import MarkerWindow
from .base_plugin import CalibrationChoreographyPlugin, ChoreographyMode, ChoreographyAction
from gaze_mapping import Gazer2D_v1x


logger = logging.getLogger(__name__)


class ScreenMarkerChoreographyPlugin(CalibrationChoreographyPlugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between
    """

    label = "Screen Marker Calibration Choreography"

    def supported_gazers(self):
        return [Gazer2D_v1x]  # FIXME: Provide complete list of supported gazers

    @staticmethod
    def __site_locations(mode: ChoreographyMode, is_2d: bool, is_3d: bool) -> list:
        assert is_2d != is_3d  # sanity check
        return [
            (0.5, 0.5),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
        ]

    def __init__(
        self,
        g_pool,
        fullscreen=True,
        marker_scale=1.0,
        sample_duration=40,
        monitor_name=None,
    ):
        super().__init__(g_pool)

        self.active_site = None
        self.sites = []

        self.fullscreen = fullscreen
        self.__marker_window = MarkerWindow(
            marker_scale=marker_scale,
            sample_duration=sample_duration,
        )
        self.__marker_window.add_observer("on_did_close", self.__on_window_did_close)

        self.__ui_selector_monitor_setter(monitor_name)

        self.circle_tracker = CircleTracker()
        self.markers = []

    def get_init_dict(self):
        d = {}
        d["fullscreen"] = self.fullscreen
        d["marker_scale"] = self.__marker_window.marker_scale
        d["monitor_name"] = self.monitor_name
        return d

    def init_ui(self):

        desc_text = ui.Info_Text("Calibrate gaze parameters using a screen based animation.")

        self.__ui_selector_monitor = ui.Selector(
            "monitor",
            label="Monitor",
            labels=self.__ui_selector_monitor_labels(),
            selection=self.__ui_selector_monitor_selection(),
            getter=self.__ui_selector_monitor_getter,
            setter=self.__ui_selector_monitor_setter,
        )

        self.__ui_switch_is_fullscreen = ui.Switch(
            "fullscreen",
            self,
            label="Use fullscreen"
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
            "sample_duration",
            self.__marker_window,
            label="Sample duration",
            min=10,
            max=100,
            step=1,
        )

        super().init_ui()
        self.menu.append(desc_text)
        self.menu.append(self.__ui_selector_monitor)
        self.menu.append(self.__ui_switch_is_fullscreen)
        self.menu.append(self.__ui_slider_marker_scale)
        self.menu.append(self.__ui_slider_sample_duration)

    def deinit_ui(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.is_active:
            self.stop()
        self.__marker_window.close()
        super().deinit_ui()

    # Private - UI

    def __ui_selector_monitor_labels(self) -> T.List[str]:
        return list(GUIMonitor.currently_connected_monitors_by_name().keys())

    def __ui_selector_monitor_selection(self) -> T.List[str]:
        return list(GUIMonitor.currently_connected_monitors_by_name().keys())

    def __ui_selector_monitor_getter(self) -> str:
        if self.monitor_name not in GUIMonitor.currently_connected_monitors_by_name():
            old_name = self.monitor_name
            new_name = GUIMonitor.primary_monitor().name
            logger.warning(f"Monitor \"{old_name}\" no longer availalbe using \"{new_name}\"")
            self.monitor_name = new_name
        return self.monitor_name

    def __ui_selector_monitor_setter(self, monitor_name: str):
        self.monitor_name = monitor_name
        if self.monitor_name not in GUIMonitor.currently_connected_monitors_by_name():
            old_name = self.monitor_name
            new_name = GUIMonitor.primary_monitor().name
            logger.warning(f"Monitor \"{old_name}\" no longer availalbe using \"{new_name}\"")
            self.monitor_name = new_name

    def recent_events(self, events):
        frame = events.get("frame")

        if not self.is_active or not frame:
            return

        gray_img = frame.gray

        # Update the marker
        self.markers = self.circle_tracker.update(gray_img)
        # Screen marker takes only Ref marker
        self.markers = [
            marker for marker in self.markers if marker["marker_type"] == "Ref"
        ]

        if len(self.markers):
            # Set the pos to be the center of the first detected marker
            marker_screen_position = self.markers[0]["img_pos"]
            marker_normal_position = self.markers[0]["norm_pos"]
        else:
            marker_screen_position = None
            marker_normal_position = None  # indicate that no reference is detected

        # Check if there are more than one markers
        if len(self.markers) > 1:
            logger.warning(f"{len(self.markers)} markers detected. Please remove all the other markers")

        # FIXME: Move to marker_window
        self.__marker_window._marker_state_valid_range = range(
            self.__marker_window._MARKER_ANIMATION_LEAD_IN_FRAMES+1,
            self.__marker_window._MARKER_ANIMATION_LEAD_IN_FRAMES + self.__marker_window.sample_duration
        )
        self.__marker_window._marker_state_animation_range = range(
            0,
            self.__marker_window.sample_duration + self.__marker_window._MARKER_ANIMATION_LEAD_IN_FRAMES + self.__marker_window._MARKER_ANIMATION_LEAD_OUT_FRAMES
        )

        # only save a valid ref position if within sample window of calibration routine
        on_position = (self.__marker_window.marker_state in self.__marker_window._marker_state_valid_range) #FIXME: Move on_position calculation to marker window

        if on_position and len(self.markers):
            ref = {}
            ref["norm_pos"] = marker_normal_position
            ref["screen_pos"] = marker_screen_position
            ref["timestamp"] = frame.timestamp
            self.ref_list.append(ref)

        # Always save pupil positions
        self.pupil_list.extend(events["pupil"])

        # TODO: This seems redundant, since on_position makes sure the marker_state is within the valid range
        if on_position and len(self.markers):
            #FIXME: Move marker_state calculation to marker window
            # self.__marker_window.marker_state = max(self.__marker_window.marker_state, self.__marker_window._marker_state_valid_range.start)
            self.__marker_window.marker_state = min(self.__marker_window.marker_state, self.__marker_window._marker_state_valid_range.stop-1)

        # Animate the screen marker
        if (self.__marker_window.marker_state in self.__marker_window._marker_state_animation_range): #FIXME: Move animation code to marker window
            if len(self.markers) or not on_position:
                self.__marker_window.marker_state += 1  # TODO: Move to marker_window as marker_state_increment()
        else:
            self.__marker_window.marker_state = self.__marker_window._marker_state_animation_range.start  #FIXME: Move to marker_window as marker_state_reset()
            if self.sites:
                self.active_site = self.sites.pop(0)
                logger.debug(f"Moving screen marker to site at {self.active_site}")
            else:
                self.stop()
                return

        # use np.arrays for per element wise math
        # self.display_pos = np.array(self.active_site)
        self.__marker_window.marker_point = np.array(self.active_site)
        self.__marker_window.gl_display()

        self.current_mode_ui_button.status_text = "{}".format(self.active_site)

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

        for marker in self.markers:
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
            if len(self.markers) > 1:
                draw_polyline(
                    pts, 1, RGBA(1.0, 0.0, 0.0, 0.5), line_type=gl.GL_POLYGON
                )

    def start(self):
        if not self.g_pool.capture.online:
            logger.error(f"{self.current_mode.label} requiers world capture video input.")
            return

        logger.info(f"Starting {self.current_mode.label}")

        self.sites = self.__site_locations(
            mode=self.current_mode,
            is_2d=self.g_pool.detection_mapping_mode == "2d",
            is_3d=self.g_pool.detection_mapping_mode == "3d",
        )
        self.active_site = self.sites.pop(0)

        self.ref_list = []
        self.pupil_list = []

        super().start()

        self.__marker_window.open(
            title=self.current_mode.label,
            monitor_name=self.monitor_name,
            is_fullscreen=self.fullscreen,
        )

    def __on_window_did_close(self):
        # TODO: Refactor this...
        self.on_notify_all(
            ChoreographyNotification(
                mode=self.current_mode,
                action=ChoreographyAction.SHOULD_STOP,
            ).to_dict()
        )

    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class

        logger.info(f"Stopping {self.current_mode.label}")

        self.__marker_window.close()
        self.current_mode_ui_button.status_text = ""

        # TODO: This part seems redundant
        if self.current_mode == ChoreographyMode.CALIBRATION:
            self.finish_calibration(self.pupil_list, self.ref_list)
        if self.current_mode == ChoreographyMode.ACCURACY_TEST:
            self.finish_accuracy_test(self.pupil_list, self.ref_list)

        super().stop()
