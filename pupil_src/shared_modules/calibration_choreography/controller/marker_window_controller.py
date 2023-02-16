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
import functools
import logging
import typing as T

import numpy as np
import observable
import OpenGL.GL as gl
from gl_utils import draw_circle_filled_func_builder
from pyglui.cygl.utils import RGBA, draw_points
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

from .gui_monitor import GUIMonitor
from .gui_window import GUIWindow

logger = logging.getLogger(__name__)


"""
To visualize the state machine that MarkerWindowController is running,
visit https://planttext.com and use the following PlantUML script:

```
@startuml

[*] --> Closed
Closed --> Open : open_window()
Closed --> [*]
Open --> Closed

state Open {
  [*] --> Idle

  Idle --> AnimatingIn : show_marker(marker_position, should_animate=True)
  Idle --> Showing : show_marker(marker_position, should_animate=False)
  Idle --> [*] : close_window()

  AnimatingIn --> AnimatingIn
  AnimatingIn --> Showing : <animating_in_complete>
  AnimatingIn --> [*] : close_window()

  Showing --> AnimatingOut : hide_marker(should_animate=True)
  Showing --> Idle : hide_marker(should_animate=False)
  Showing --> [*] : close_window()

  AnimatingOut --> AnimatingOut
  AnimatingOut --> Idle : <animating_out_complete>
  AnimatingOut --> [*] : close_window()
}

@enduml
```
"""


class MarkerWindowController(observable.Observable):
    _CLICKS_NEEDED_TO_CLOSE = 5

    # frames of marker shown before starting to sample
    _MARKER_ANIMATION_DURATION_IN = 25
    # frames of markers shown after sampling is done
    _MARKER_ANIMATION_DURATION_OUT = 5

    _MARKER_CIRCLE_RGB_OUTER = (0.0, 0.0, 0.0)
    _MARKER_CIRCLE_RGB_MIDDLE = (1.0, 1.0, 1.0)
    _MARKER_CIRCLE_RGB_INNER = (0.0, 0.0, 0.0)
    _MARKER_CIRCLE_RGB_FEEDBACK_INVALID = (0.8, 0.0, 0.0)
    _MARKER_CIRCLE_RGB_FEEDBACK_VALID = (0.0, 0.8, 0.0)

    _MARKER_CIRCLE_SIZE_OUTER = 60
    _MARKER_CIRCLE_SIZE_MIDDLE = 38
    _MARKER_CIRCLE_SIZE_INNER = 19
    _MARKER_CIRCLE_SIZE_FEEDBACK = 3

    def __init__(self, marker_scale: float):
        # Public properties
        self.marker_scale = marker_scale
        self.is_marker_detected = False
        # Private state
        self.__state: _MarkerWindowState = MarkerWindowStateClosed()
        self.__window = GUIWindow()
        self.__window.add_observer("on_left_click", self._on_left_click)
        self.__window.add_observer("on_key_press_escape", self._on_key_press_escape)
        # Private font
        self.__glfont = fontstash.Context()
        self.__glfont.add_font("opensans", get_opensans_font_path())
        self.__glfont.set_size(32)
        self.__glfont.set_color_float((0.2, 0.5, 0.9, 1.0))
        self.__glfont.set_align_string(v_align="center")
        # Private helper
        self.__draw_circle_filled = draw_circle_filled_func_builder(cache_size=4)

    # Public - Marker Management

    def show_marker(self, marker_position: T.Tuple[float, float], should_animate: bool):
        log_ignored_call = lambda: logger.debug(
            f"show_marker called for state {type(self.__state)}; ignoring the call."
        )

        if isinstance(self.__state, MarkerWindowStateClosed):
            log_ignored_call()
            return
        if isinstance(self.__state, MarkerWindowStateOpened):
            if isinstance(self.__state, MarkerWindowStateIdle):
                if should_animate:
                    self.__state = MarkerWindowStateAnimatingInMarker(
                        marker_position=marker_position,
                        clicks_needed=self.__state.clicks_needed,
                        animation_duration=self._MARKER_ANIMATION_DURATION_IN,
                    )
                else:
                    self.__state = MarkerWindowStateShowingMarker(
                        marker_position=marker_position,
                        clicks_needed=self.__state.clicks_needed,
                    )
                return
            if isinstance(self.__state, MarkerWindowStateShowingMarker):
                log_ignored_call()
                return
            if isinstance(self.__state, MarkerWindowStateAnimatingInMarker):
                log_ignored_call()
                return
            if isinstance(self.__state, MarkerWindowStateAnimatingOutMarker):
                log_ignored_call()
                return
        raise UnhandledMarkerWindowStateError(self.__state)

    def hide_marker(self, should_animate: bool):
        log_ignored_call = lambda: logger.debug(
            f"hide_marker called for state {type(self.__state)}; ignoring the call."
        )

        if isinstance(self.__state, MarkerWindowStateClosed):
            log_ignored_call()
            return
        if isinstance(self.__state, MarkerWindowStateOpened):
            if isinstance(self.__state, MarkerWindowStateIdle):
                log_ignored_call()
                return
            if isinstance(self.__state, MarkerWindowStateShowingMarker):
                if should_animate:
                    self.__state = MarkerWindowStateAnimatingOutMarker(
                        marker_position=self.__state.marker_position,
                        clicks_needed=self.__state.clicks_needed,
                        animation_duration=self._MARKER_ANIMATION_DURATION_OUT,
                    )
                return
            if isinstance(self.__state, MarkerWindowStateAnimatingInMarker):
                log_ignored_call()
                return
            if isinstance(self.__state, MarkerWindowStateAnimatingOutMarker):
                log_ignored_call()
                return
        raise UnhandledMarkerWindowStateError(self.__state)

    # Public - Window Management

    @property
    def window_state(self) -> "_MarkerWindowState":
        return self.__state

    def open_window(self, monitor_name: str, title: str, is_fullscreen: bool):
        log_ignored_call = lambda: logger.debug(
            f"open_window called for state {type(self.__state)}; ignoring the call."
        )

        if isinstance(self.__state, MarkerWindowStateOpened):
            log_ignored_call()
            return
        if isinstance(self.__state, MarkerWindowStateClosed):
            gui_monitor = GUIMonitor.find_monitor_by_name(monitor_name)
            if gui_monitor is None:
                gui_monitor = GUIMonitor.primary_monitor()
                logger.warning(
                    f'Monitor named "{monitor_name}" no longer available. Using primary monitor "{gui_monitor.name}"'
                )
            window_size = None if is_fullscreen else (640, 360)
            self.__window.open(
                gui_monitor=gui_monitor,
                title=title,
                is_fullscreen=is_fullscreen,
                size=window_size,
            )
            self.__window.cursor_hide()
            self.__state = MarkerWindowStateIdle(
                clicks_needed=self._CLICKS_NEEDED_TO_CLOSE
            )
            return
        raise UnhandledMarkerWindowStateError(self.__state)

    def close_window(self):
        log_ignored_call = lambda: logger.debug(
            f"close_window called for state {type(self.__state)}; ignoring the call."
        )

        if isinstance(self.__state, MarkerWindowStateClosed):
            log_ignored_call()
            return

        if isinstance(self.__state, MarkerWindowStateOpened):
            self.on_window_will_close()
            self.__window.close()
            self.__state = MarkerWindowStateClosed()
            self.on_window_did_close()
            return

        raise UnhandledMarkerWindowStateError(self.__state)

    def update_state(self):
        self.__state.update_state()

        if isinstance(self.__state, MarkerWindowStateClosed):
            return  # No-op

        elif isinstance(self.__state, MarkerWindowStateIdle):
            return  # No-op

        elif isinstance(self.__state, MarkerWindowStateAnimatingInMarker):
            if self.__state.is_complete:
                self.__state = MarkerWindowStateShowingMarker(
                    marker_position=self.__state.marker_position,
                    clicks_needed=self.__state.clicks_needed,
                )

        elif isinstance(self.__state, MarkerWindowStateShowingMarker):
            return  # No-op

        elif isinstance(self.__state, MarkerWindowStateAnimatingOutMarker):
            if self.__state.is_complete:
                self.__state = MarkerWindowStateIdle(
                    clicks_needed=self.__state.clicks_needed
                )

        else:
            raise UnhandledMarkerWindowStateError(self.__state)

    def draw_window(self):
        if self.__window.window_size == (0, 0):
            # On Windows we get a window_size of (0, 0) when either minimizing the
            # Window or when tabbing out (rendered only in the background). We get
            # errors when we call the code below with window size (0, 0). Anyways we
            # probably want to stop calibration in this case as it will screw up the
            # calibration anyways.
            self.close_window()
            return

        if isinstance(self.__state, MarkerWindowStateClosed):
            return

        elif isinstance(self.__state, MarkerWindowStateOpened):
            clicks_needed = self.__state.clicks_needed
            marker_position = self.__state.marker_position
            marker_alpha = self.__state.marker_color_alpha
            is_valid = (not self.__state.is_animating) and self.is_marker_detected

            if clicks_needed == 0:
                self.close_window()
                return

            with self.__window.drawing_context() as gl_context:
                if gl_context:
                    self.__draw_circle_marker(
                        position=marker_position, is_valid=is_valid, alpha=marker_alpha
                    )
                    self.__draw_status_text(clicks_needed=clicks_needed)

        else:
            raise UnhandledMarkerWindowStateError(self.__state)

    # Public - Callbacks

    def on_window_will_close(self):
        pass

    def on_window_did_close(self):
        pass

    # Private

    def _on_left_click(self):
        if isinstance(self.__state, MarkerWindowStateOpened):
            self.__state.clicks_needed -= 1

    def _on_key_press_escape(self):
        if isinstance(self.__state, MarkerWindowStateOpened):
            self.__state.clicks_needed = 0

    def __draw_circle_marker(
        self, position: T.Optional[T.Tuple[float, float]], is_valid: bool, alpha: float
    ):
        if position is None:
            return

        radius = self.__marker_radius
        screen_point = self.__marker_position_on_screen(position)

        if is_valid:
            marker_circle_rgb_feedback = self._MARKER_CIRCLE_RGB_FEEDBACK_VALID
        else:
            marker_circle_rgb_feedback = self._MARKER_CIRCLE_RGB_FEEDBACK_INVALID

        # TODO: adjust size such that they correspond to old marker sizes
        # TODO: adjust num_points such that circles look smooth; smaller circles need less points
        # TODO: compare runtimes with `draw_points`

        self.__draw_circle_filled(
            screen_point,
            size=self._MARKER_CIRCLE_SIZE_OUTER * radius,
            color=RGBA(*self._MARKER_CIRCLE_RGB_OUTER, alpha),
        )
        self.__draw_circle_filled(
            screen_point,
            size=self._MARKER_CIRCLE_SIZE_MIDDLE * radius,
            color=RGBA(*self._MARKER_CIRCLE_RGB_MIDDLE, alpha),
        )
        self.__draw_circle_filled(
            screen_point,
            size=self._MARKER_CIRCLE_SIZE_INNER * radius,
            color=RGBA(*self._MARKER_CIRCLE_RGB_INNER, alpha),
        )
        self.__draw_circle_filled(
            screen_point,
            size=self._MARKER_CIRCLE_SIZE_FEEDBACK * radius,
            color=RGBA(*marker_circle_rgb_feedback, alpha),
        )

    def __draw_status_text(self, clicks_needed: int):
        if clicks_needed >= self._CLICKS_NEEDED_TO_CLOSE:
            return
        window_size = self.__window.window_size
        closing_text = f"Touch {clicks_needed} more times to cancel."
        self.__glfont.set_size(int(window_size[0] / 30.0))
        self.__glfont.draw_text(
            window_size[0] / 2.0, window_size[1] / 4.0, closing_text
        )

    @property
    def __marker_radius(self) -> float:
        return self.marker_scale * self.__window.content_scale

    def __marker_position_on_screen(self, marker_position) -> T.Tuple[float, float]:
        padding = 90 * self.__marker_radius
        window_size = self.__window.window_size
        return (
            _map_value(
                marker_position[0], out_range=(padding, window_size[0] - padding)
            ),
            _map_value(
                marker_position[1], out_range=(window_size[1] - padding, padding)
            ),
        )


### MARKER WINDOW STATES


class _MarkerWindowState(abc.ABC):
    def update_state(self):
        pass

    def _repr_items(self):
        return []

    def __repr__(self):
        items = self._repr_items()
        items_str = (" " + " ".join(items)) if items else ""
        return f"<{self.__class__.__name__}{items_str}>"


class MarkerWindowStateClosed(_MarkerWindowState):
    pass


class MarkerWindowStateOpened(_MarkerWindowState, abc.ABC):
    def __init__(
        self, marker_position: T.Optional[T.Tuple[float, float]], clicks_needed: int
    ):
        self.clicks_needed = clicks_needed
        self.__marker_position = marker_position
        self.__total_clicks_needed = clicks_needed

    @property
    def marker_position(self) -> T.Optional[T.Tuple[float, float]]:
        return self.__marker_position

    @property
    @abc.abstractmethod
    def is_animating(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def marker_color_alpha(self) -> float:
        pass

    def _repr_items(self):
        return [
            f"clicks_needed={self.clicks_needed}/{self.__total_clicks_needed}"
        ] + super()._repr_items()


class MarkerWindowStateIdle(MarkerWindowStateOpened):
    def __init__(self, clicks_needed: int):
        super().__init__(marker_position=None, clicks_needed=clicks_needed)

    @property
    def is_animating(self) -> bool:
        return False

    @property
    def marker_color_alpha(self) -> float:
        return 0.0


class MarkerWindowStateAnimatingMarker(MarkerWindowStateOpened, abc.ABC):
    def __init__(
        self,
        marker_position: T.Tuple[float, float],
        clicks_needed: int,
        animation_duration: int,
    ):
        super().__init__(marker_position=marker_position, clicks_needed=clicks_needed)
        self.__current_duration = 0
        self.__animation_duration = animation_duration

    @property
    def is_animating(self) -> bool:
        return True

    @property
    def is_complete(self) -> bool:
        return self.__current_duration == self.__animation_duration

    @property
    def progress(self) -> float:
        if self.__animation_duration > 0:
            return self.__current_duration / self.__animation_duration
        else:
            return 1.0  # For cases where animation duration is 0.

    def update_state(self):
        super().update_state()
        if self.__current_duration < self.__animation_duration:
            self.__current_duration += 1

    def _repr_items(self):
        return [
            f"{self.marker_position}",
            f"animation={self.__current_duration}/{self.__animation_duration}",
        ] + super()._repr_items()


class MarkerWindowStateAnimatingInMarker(MarkerWindowStateAnimatingMarker):
    @property
    def marker_color_alpha(self) -> float:
        return self.progress


class MarkerWindowStateShowingMarker(MarkerWindowStateOpened):
    def __init__(self, marker_position: T.Tuple[float, float], clicks_needed: int):
        super().__init__(marker_position=marker_position, clicks_needed=clicks_needed)

    @property
    def is_animating(self) -> bool:
        return False

    @property
    def marker_color_alpha(self) -> float:
        return 1.0

    def _repr_items(self):
        return [f"{self.marker_position}"] + super()._repr_items()


class MarkerWindowStateAnimatingOutMarker(MarkerWindowStateAnimatingMarker):
    @property
    def marker_color_alpha(self) -> float:
        return 1.0 - self.progress


class UnhandledMarkerWindowStateError(NotImplementedError):
    def __init__(self, state: _MarkerWindowState):
        super().__init__(f"Unhandled marker window state: {type(state)}")


### PRIVATE HELPER FUNCTIONS


def _map_value(value, in_range=(0, 1), out_range=(0, 1)):
    ratio = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
    return (value - in_range[0]) * ratio + out_range[0]


# easing functions for animation of the marker fade in/out
def _easeInOutQuad(t, b, c, d):
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


def _interp_fn(t, b, c, d, start_sample=15.0, stop_sample=55.0):
    # ease in, sample, ease out
    if t < start_sample:
        return _easeInOutQuad(t, b, c, start_sample)
    elif t > stop_sample:
        return 1 - _easeInOutQuad(t - stop_sample, b, c, d - stop_sample)
    else:
        return 1.0
