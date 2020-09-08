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
            monitors = glfw.glfwGetMonitors()
            if any(_is_point_within_monitor(m, previous_position) for m in monitors):
                return previous_position
            else:
                return default_position

        else:
            raise NotImplementedError(f"Unsupported system: {os_name}")


def _is_point_within_monitor(monitor, point) -> bool:
    x, y, w, h = glfw.glfwGetMonitorWorkarea(monitor)

    is_within_horizontally = x <= point[0] < x + w
    is_within_vertically = y <= point[1] < y + h

    return is_within_horizontally and is_within_vertically
