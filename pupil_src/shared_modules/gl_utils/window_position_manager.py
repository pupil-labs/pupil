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
import platform
import typing as T

import gl_utils
import glfw

from .utils import GLFWErrorReporting

GLFWErrorReporting.set_default()


class WindowPositionManager:
    def __init__(self):
        pass

    @staticmethod
    def new_window_position(
        window,
        default_position: T.Tuple[int, int],
        previous_position: T.Optional[T.Tuple[int, int]],
    ) -> T.Tuple[int, int]:
        if previous_position is None:
            return default_position

        os_name = platform.system()

        if os_name == "Darwin":
            # The OS handle re-positioning windows with invalid positions
            return previous_position

        elif os_name == "Linux":
            # The OS handle re-positioning windows with invalid positions
            return previous_position

        elif os_name == "Windows":

            def validate_previous_position(monitor) -> bool:
                return _will_window_be_visible_in_monitor(
                    window=window,
                    monitor=monitor,
                    window_position=previous_position,
                )

            if any(validate_previous_position(m) for m in glfw.get_monitors()):
                return previous_position
            else:
                return default_position

        else:
            raise NotImplementedError(f"Unsupported system: {os_name}")


def _will_window_be_visible_in_monitor(
    window, monitor, window_position, min_visible_width=30, min_visible_height=20
) -> bool:
    # Get the current window size and edges, and monitor rect
    window_size = glfw.get_window_size(window)
    window_edges = gl_utils.get_window_frame_size_margins(window)
    monitor_rect = gl_utils.get_monitor_workarea_rect(monitor)

    # Calculate what the title bar rect would be
    # if the proposed `window_position` would be the actual window position
    title_bar_rect = gl_utils._Rectangle(
        x=window_position[0] - window_edges.left,
        y=window_position[1] - window_edges.top,
        width=window_size[0] + window_edges.left + window_edges.right,
        height=window_edges.top,
    )

    # Calculate the part of the title bar that is visible in the monitor, if any
    visible_rect = title_bar_rect.intersection(monitor_rect)

    # Return true if the visible title bar rect is big enough
    return (
        visible_rect is not None
        and min_visible_width <= visible_rect.width
        and min_visible_height <= visible_rect.height
    )
