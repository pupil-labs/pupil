import typing as T

import OpenGL.GL as gl
import glfw
from gl_utils import clear_gl_screen

from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from pyglui.cygl.utils import draw_points
from pyglui.cygl.utils import RGBA

from .gui_monitor import GUIMonitor
from .gui_window import GUIWindow


class MarkerWindow(GUIWindow):

    _CLICKS_TO_CLOSE = 5

    _MARKER_CIRCLE_RGB_OUTER = (0.0, 0.0, 0.0)
    _MARKER_CIRCLE_RGB_MIDDLE = (1.0, 1.0, 1.0)
    _MARKER_CIRCLE_RGB_INNER = (0.0, 0.0, 0.0)
    _MARKER_CIRCLE_RGB_VALID = (0.0, 0.8, 0.0)
    _MARKER_CIRCLE_RGB_INVALID = (0.8, 0.0, 0.0)

    _MARKER_CIRCLE_SIZE_OUTER = 60
    _MARKER_CIRCLE_SIZE_MIDDLE = 38
    _MARKER_CIRCLE_SIZE_INNER = 19
    _MARKER_CIRCLE_SIZE_FEEDBACK = 3

    _MARKER_CIRCLE_SHARPNESS_OUTER = 0.9
    _MARKER_CIRCLE_SHARPNESS_MIDDLE = 0.8
    _MARKER_CIRCLE_SHARPNESS_INNER = 0.55
    _MARKER_CIRCLE_SHARPNESS_FEEDBACK = 0.5

    _MARKER_ANIMATION_LEAD_IN_FRAMES = 25  # frames of marker shown before starting to sample
    _MARKER_ANIMATION_LEAD_OUT_FRAMES = 5  # frames of markers shown after sampling is done

    def __init__(self, marker_scale: float = 1.0, sample_duration: int = 40):
        super().__init__()
        self.__reset_state()

        # Externally updatable params
        self.marker_scale = marker_scale
        self.sample_duration = sample_duration  # number of frames to sample per site

        # Internal font params
        self.__glfont = fontstash.Context()
        self.__glfont.add_font("opensans", get_opensans_font_path())
        self.__glfont.set_size(32)
        self.__glfont.set_color_float((0.2, 0.5, 0.9, 1.0))
        self.__glfont.set_align_string(v_align="center")

    def open(self, monitor_name: str, title: str, is_fullscreen: bool):
        gui_monitor = GUIMonitor.find_monitor_by_name(monitor_name)
        if gui_monitor is None:
            # TODO: Log warning that the primary monitor is used
            gui_monitor = GUIMonitor.primary_monitor()

        self.__reset_state()

        if is_fullscreen:
            super().open(
                gui_monitor=gui_monitor,
                title=title,
                is_fullscreen=True,
                size=None,
            )
        else:
            super().open(
                gui_monitor=gui_monitor,
                title=title,
                is_fullscreen=False,
                size=(640, 360)
            )

        # # This makes it harder to accidentally tab out of fullscreen by clicking on
        # # some other window (e.g. when having two monitors). On the other hand you
        # # want a cursor to adjust the window size when not in fullscreen mode.
        # cursor = GLFW_CURSOR_DISABLED if is_fullscreen else GLFW_CURSOR_HIDDEN
        # glfwSetInputMode(self.__gl_handle, GLFW_CURSOR, cursor)

    def gl_display(self):
        super().gl_display()

        if self.__current_clicks_to_close <= 0:
            self.close()
            return

        if self.window_size == (0, 0):
            # On Windows we get a window_size of (0, 0) when either minimizing the
            # Window or when tabbing out (rendered only in the background). We get
            # errors when we call the code below with window size (0, 0). Anyways we
            # probably want to stop calibration in this case as it will screw up the
            # calibration anyways.
            self.close()
            return

        with self._switch_to_current_context():
            clear_gl_screen()

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, self.window_size[0], self.window_size[1], 0, -1, 1)
            # Switch back to Model View Matrix
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            self.__gl_draw_marker()
            self.__gl_draw_closing_text()

            glfw.glfwSwapBuffers(self.unsafe_handle)

    def on_key(self, gl_handle, key, scancode, action, mods):
        super().on_key(gl_handle, key, scancode, action, mods)
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_ESCAPE:
                self.__current_clicks_to_close = 0

    def on_mouse_button(self, gl_handle, button, action, mods):
        super().on_mouse_button(gl_handle, button, action, mods)
        if action == glfw.GLFW_PRESS:
            self.__current_clicks_to_close -= 1

    # Private

    def __reset_state(self):
        # Externally updateable state
        self.marker_state = 0.0
        self.marker_point = (-1, -1)
        self.is_detection_valid = False

        # Internally updatable state
        self.__current_clicks_to_close = self._CLICKS_TO_CLOSE

    def __gl_draw_marker(self):
        r2 = 2 * self.__current_marker_radius
        alpha = self.__current_marker_color_alpha
        points = [self.__current_marker_position]

        if self.is_detection_valid:
            marker_circle_rgb_feedback = self._MARKER_CIRCLE_RGB_VALID
        else:
            marker_circle_rgb_feedback = self._MARKER_CIRCLE_RGB_INVALID

        draw_points(
            points,
            size=self._MARKER_CIRCLE_SIZE_OUTER * r2,
            color=RGBA(*self._MARKER_CIRCLE_RGB_OUTER, alpha),
            sharpness=self._MARKER_CIRCLE_SHARPNESS_OUTER,
        )
        draw_points(
            points,
            size=self._MARKER_CIRCLE_SIZE_MIDDLE * r2,
            color=RGBA(*self._MARKER_CIRCLE_RGB_MIDDLE, alpha),
            sharpness=self._MARKER_CIRCLE_SHARPNESS_MIDDLE,
        )
        draw_points(
            points,
            size=self._MARKER_CIRCLE_SIZE_INNER * r2,
            color=RGBA(*self._MARKER_CIRCLE_RGB_INNER, alpha),
            sharpness=self._MARKER_CIRCLE_SHARPNESS_INNER,
        )
        draw_points(
            points,
            size=self._MARKER_CIRCLE_SIZE_FEEDBACK * r2,
            color=RGBA(*marker_circle_rgb_feedback, alpha),
            sharpness=self._MARKER_CIRCLE_SHARPNESS_FEEDBACK,
        )

    def __gl_draw_closing_text(self):
        window_width, window_height = self.window_size
        closing_text = f"Touch {self.__current_clicks_to_close} more times to cancel."
        if self.__current_clicks_to_close < self._CLICKS_TO_CLOSE:
            self.__glfont.set_size(int(window_width / 30.0))
            self.__glfont.draw_text(
                window_width / 2.0,
                window_height / 4.0,
                closing_text
            )

    @property
    def __current_marker_position(self) -> T.Tuple[float, float]:
        padding = 90 * self.__current_marker_radius
        return (
            map_value(self.marker_point[0], out_range=(padding, self.window_size[0] - padding)),
            map_value(self.marker_point[1], out_range=(self.window_size[1] - padding, padding)),
        )

    @property
    def __current_marker_radius(self) -> float:
        return self.marker_scale * self.hdpi_factor

    @property
    def __current_marker_color_alpha(self) -> float:
        return interp_fn(
            self.marker_state,
            0.0,
            1.0,
            float(self.sample_duration + self._MARKER_ANIMATION_LEAD_IN_FRAMES + self._MARKER_ANIMATION_LEAD_OUT_FRAMES),
            float(self._MARKER_ANIMATION_LEAD_IN_FRAMES),
            float(self.sample_duration + self._MARKER_ANIMATION_LEAD_IN_FRAMES),
        )


def map_value(value, in_range=(0, 1), out_range=(0, 1)):
    ratio = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
    return (value - in_range[0]) * ratio + out_range[0]


# easing functions for animation of the marker fade in/out
def easeInOutQuad(t, b, c, d):
    """Robert Penner easing function examples at: http://gizma.com/easing/
    t = current time in frames or whatever unit
    b = beginning/start value
    c = change in value
    d = duration

    """
    t /= d / 2
    if t < 1:
        return c / 2 * t * t + b
    t -= 1
    return -c / 2 * (t * (t - 2) - 1) + b


def interp_fn(t, b, c, d, start_sample=15.0, stop_sample=55.0):
    # ease in, sample, ease out
    if t < start_sample:
        return easeInOutQuad(t, b, c, start_sample)
    elif t > stop_sample:
        return 1 - easeInOutQuad(t - stop_sample, b, c, d - stop_sample)
    else:
        return 1.0
