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
import platform
import typing as T

import glfw


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

            if any(validate_previous_position(m) for m in glfw.glfwGetMonitors()):
                return previous_position
            else:
                return default_position

        else:
            raise NotImplementedError(f"Unsupported system: {os_name}")


def _will_window_be_visible_in_monitor(
    window, monitor, window_position, min_visible_width=10, min_visible_height=5
) -> bool:
    monitor_rect = glfw.glfwGetMonitorWorkarea(monitor)
    title_bar_rect = glfw.get_window_title_bar_rect(window)
    visible_rect = glfw.rectangle_intersection(monitor_rect, title_bar_rect)
    return (
        visible_rect is not None
        and min_visible_width <= visible_rect.width
        and min_visible_height <= visible_rect.height
    )


