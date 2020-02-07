# """
# (*)~---------------------------------------------------------------------------
# Pupil - eye tracking platform
# Copyright (C) 2012-2020 Pupil Labs

# Distributed under the terms of the GNU
# Lesser General Public License (LGPL v3.0).
# See COPYING and COPYING.LESSER for license details.
# ---------------------------------------------------------------------------~(*)
# """
import logging
from platform import system

import cv2
import numpy as np

import audio
from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_polyline, RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from plugin import Plugin
from circle_detector import CircleTracker
from gl_utils import adjust_gl_view, clear_gl_screen, basic_gl_setup
import OpenGL.GL as gl
from glfw import *

from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyMode,
    ChoreographyAction,
)
from gaze_mapping import Gazer2D_v1x


logger = logging.getLogger(__name__)


class SingleMarkerChoreographyPlugin(CalibrationChoreographyPlugin):
    """Calibrate using a single marker.
       Move your head for example in a spiral motion while gazing
       at the marker to quickly sample a wide range gaze angles.
    """

    label = "Single Marker Calibration Choreography"

    def supported_gazers(self):
        return [Gazer2D_v1x]  # FIXME: Provide complete list of supported gazers

    def __init__(
        self,
        g_pool,
        marker_mode="Full screen",
        marker_scale=1.0,
        sample_duration=40,
        monitor_idx=0,
    ):
        super().__init__(g_pool)
        self.screen_marker_state = 0.0
        self.lead_in = 25  # frames of marker shown before starting to sample

        self.display_pos = (0.5, 0.5)
        self.on_position = False
        self.pos = None

        self.marker_scale = marker_scale

        self._window = None

        self.menu = None

        self.stop_marker_found = False
        self.auto_stop = 0
        self.auto_stop_max = 30

        self.monitor_idx = monitor_idx
        self.marker_mode = marker_mode  # TODO: Create an enum for marker_mode
        self.clicks_to_close = 5

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))
        self.glfont.set_align_string(v_align="center")

        # UI Platform tweaks
        if system() == "Linux":
            self.window_position_default = (0, 0)
        elif system() == "Windows":
            self.window_position_default = (8, 90)
        else:
            self.window_position_default = (0, 0)

        self.circle_tracker = CircleTracker()
        self.markers = []

    def get_init_dict(self):
        d = {}
        d["marker_mode"] = self.marker_mode
        d["marker_scale"] = self.marker_scale
        d["monitor_idx"] = self.monitor_idx
        return d

    def cleanup(self):
        super().cleanup()

    def init_ui(self):
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]

        def get_monitors_idx_list():
            monitors = [glfwGetMonitorName(m) for m in glfwGetMonitors()]
            return range(len(monitors)), monitors

        if self.monitor_idx not in get_monitors_idx_list()[0]:
            logger.warning(
                f"Monitor at index {self.monitor_idx} no longer availalbe using default"
            )
            self.monitor_idx = 0

        desc_text = ui.Info_Text(
            "Calibrate using a single marker. Gaze at the center of the marker and move your head (e.g. in a slow spiral movement). This calibration method enables you to quickly sample a wide range of gaze angles and cover a large range of your FOV."
        )

        self.__ui_selector_marker_mode = ui.Selector(
            "marker_mode",
            self,
            selection=["Full screen", "Window", "Manual"],
            label="Marker display mode",
        )

        # TODO: potential race condition through selection_getter. Should ensure
        # that current selection will always be present in the list returned by the
        # selection_getter. Highly unlikely though as this needs to happen between
        # having clicked the Selector and the next redraw.
        # See https://github.com/pupil-labs/pyglui/pull/112/commits/587818e9556f14bfedd8ff8d093107358745c29b
        self.__ui_selector_monitor_idx = ui.Selector(
            "monitor_idx", self, selection_getter=get_monitors_idx_list, label="Monitor"
        )

        self.__ui_selector_marker_scale = ui.Slider(
            "marker_scale", self, step=0.1, min=0.5, max=2.0, label="Marker size"
        )

        super().init_ui()
        self.menu.append(desc_text)
        self.menu.append(self.__ui_selector_marker_mode)
        self.menu.append(self.__ui_selector_monitor_idx)
        self.menu.append(self.__ui_selector_marker_scale)

    def deinit_ui(self):
        if self.is_active:
            self.stop()
        if self._window:
            self.close_window()
        super().deinit_ui()

    def start(self):
        if not self.g_pool.capture.online:
            logger.error("This calibration requires world capture video input.")
            return
        super().start()
        audio.say(f"Starting {self.current_mode.label}")
        logger.info(f"Starting {self.current_mode.label}")

        self.ref_list = []
        self.pupil_list = []
        self.clicks_to_close = 5

        if self.marker_mode != "Manual":
            self.open_window(self.current_mode.label)

    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class
        audio.say(f"Stopping  {self.current_mode.label}")
        logger.info(f"Stopping  {self.current_mode.label}")
        self.smooth_pos = 0, 0
        self.counter = 0
        self.close_window()
        self.current_mode_ui_button.status_text = ""
        if self.current_mode == ChoreographyMode.CALIBRATION:
            self.finish_calibration(self.pupil_list, self.ref_list)
        if self.current_mode == ChoreographyMode.ACCURACY_TEST:
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def open_window(self, title="new_window"):
        if not self._window:
            if self.marker_mode == "Full screen":
                try:
                    monitor = glfwGetMonitors()[self.monitor_idx]
                except Exception:
                    logger.warning(
                        "Monitor at index %s no longer availalbe using default" % idx
                    )
                    self.monitor_idx = 0
                    monitor = glfwGetMonitors()[self.monitor_idx]
                (
                    width,
                    height,
                    redBits,
                    blueBits,
                    greenBits,
                    refreshRate,
                ) = glfwGetVideoMode(monitor)
            else:
                monitor = None
                width, height = 640, 360

            self._window = glfwCreateWindow(
                width, height, title, monitor=monitor, share=glfwGetCurrentContext()
            )
            if self.marker_mode == "Window":
                glfwSetWindowPos(
                    self._window,
                    self.window_position_default[0],
                    self.window_position_default[1],
                )

            glfwSetInputMode(self._window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN)

            # Register callbacks
            glfwSetFramebufferSizeCallback(self._window, on_resize)
            glfwSetKeyCallback(self._window, self.on_window_key)
            glfwSetMouseButtonCallback(self._window, self.on_window_mouse_button)
            on_resize(self._window, *glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

    def close_window(self):
        if self._window:
            # enable mouse display
            active_window = glfwGetCurrentContext()
            glfwSetInputMode(self._window, GLFW_CURSOR, GLFW_CURSOR_NORMAL)
            glfwDestroyWindow(self._window)
            self._window = None
            glfwMakeContextCurrent(active_window)

    def on_window_key(self, window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if self.current_mode == ChoreographyMode.CALIBRATION:
                target_key = GLFW_KEY_C
            if self.current_mode == ChoreographyMode.ACCURACY_TEST:
                target_key = GLFW_KEY_T
            if key == GLFW_KEY_ESCAPE or key == target_key:
                self.clicks_to_close = 0

    def on_window_mouse_button(self, window, button, action, mods):
        if action == GLFW_PRESS:
            self.clicks_to_close -= 1

    def recent_events(self, events):
        frame = events.get("frame")
        if self.is_active and frame:
            gray_img = frame.gray

            if self.clicks_to_close <= 0:
                self.stop()
                return

            # Update the marker
            self.markers = self.circle_tracker.update(gray_img)

            self.stop_marker_found = False
            if len(self.markers):
                # Set the pos to be the center of the first detected marker
                marker_pos = self.markers[0]["img_pos"]
                self.pos = self.markers[0]["norm_pos"]
                # Check if there are stop markers
                for marker in self.markers:
                    if marker["marker_type"] == "Stop":
                        self.auto_stop += 1
                        self.stop_marker_found = True
                        break
            else:
                self.pos = None  # indicate that no reference is detected

            if self.stop_marker_found is False:
                self.auto_stop = 0

            # Check if there are more than one markers
            if len(self.markers) > 1:
                audio.tink()
                logger.warning(
                    "{} markers detected. Please remove all the other markers".format(
                        len(self.markers)
                    )
                )

            # only save a valid ref position if within sample window of calibraiton routine
            on_position = self.lead_in < self.screen_marker_state

            if on_position and len(self.markers) and not self.stop_marker_found:
                ref = {}
                ref["norm_pos"] = self.pos
                ref["screen_pos"] = marker_pos
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)

            # always save pupil positions
            self.pupil_list.extend(events["pupil"])

            # Animate the screen marker
            if len(self.markers) or not on_position:
                self.screen_marker_state += 1

            # Stop if autostop condition is satisfied:
            if self.auto_stop >= self.auto_stop_max:
                self.auto_stop = 0
                self.stop()

            # use np.arrays for per element wise math
            self.on_position = on_position
        if self._window:
            self.gl_display_in_window()

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.is_active:
            # draw the largest ellipse of all detected markers
            for marker in self.markers:
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
                if len(self.markers) > 1:
                    draw_polyline(
                        pts, 1, RGBA(1.0, 0.0, 0.0, 0.5), line_type=gl.GL_POLYGON
                    )

            # draw indicator on the stop marker(s)
            if self.auto_stop:
                for marker in self.markers:
                    if marker["marker_type"] == "Stop":
                        e = marker["ellipses"][-1]
                        pts = cv2.ellipse2Poly(
                            (int(e[0][0]), int(e[0][1])),
                            (int(e[1][0] / 2), int(e[1][1] / 2)),
                            int(e[-1]),
                            0,
                            360,
                            360 // self.auto_stop_max,
                        )
                        indicator = [e[0]] + pts[self.auto_stop :].tolist() + [e[0]]
                        draw_polyline(
                            indicator,
                            color=RGBA(8.0, 0.1, 0.1, 0.8),
                            line_type=gl.GL_POLYGON,
                        )

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        if glfwWindowShouldClose(self._window):
            self.close_window()
            return

        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        hdpi_factor = getHDPIFactor(self._window)
        r = self.marker_scale * hdpi_factor
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetFramebufferSize(self._window)
        gl.glOrtho(0, p_window_size[0], p_window_size[1], 0, -1, 1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        def map_value(value, in_range=(0, 1), out_range=(0, 1)):
            ratio = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
            return (value - in_range[0]) * ratio + out_range[0]

        pad = 90 * r
        screen_pos = (
            map_value(self.display_pos[0], out_range=(pad, p_window_size[0] - pad)),
            map_value(self.display_pos[1], out_range=(p_window_size[1] - pad, pad)),
        )
        alpha = (
            1.0
        )  # interp_fn(self.screen_marker_state,0.,1.,float(self.sample_duration+self.lead_in+self.lead_out),float(self.lead_in),float(self.sample_duration+self.lead_in))

        r2 = 2 * r
        draw_points(
            [screen_pos], size=60 * r2, color=RGBA(0.0, 0.0, 0.0, alpha), sharpness=0.9
        )
        draw_points(
            [screen_pos], size=38 * r2, color=RGBA(1.0, 1.0, 1.0, alpha), sharpness=0.8
        )
        draw_points(
            [screen_pos], size=19 * r2, color=RGBA(0.0, 0.0, 0.0, alpha), sharpness=0.55
        )

        # some feedback on the detection state
        color = (
            RGBA(0.0, 0.8, 0.0, alpha)
            if len(self.markers) and self.on_position
            else RGBA(0.8, 0.0, 0.0, alpha)
        )
        draw_points([screen_pos], size=3 * r2, color=color, sharpness=0.5)

        if self.clicks_to_close < 5:
            self.glfont.set_size(int(p_window_size[0] / 30.0))
            self.glfont.draw_text(
                p_window_size[0] / 2.0,
                p_window_size[1] / 4.0,
                "Touch {} more times to cancel calibration.".format(
                    self.clicks_to_close
                ),
            )

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


# window calbacks
def on_resize(window, w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w, h)
    glfwMakeContextCurrent(active_window)
