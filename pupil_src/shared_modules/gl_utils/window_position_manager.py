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
            return previous_position
        elif os_name == "Linux":
            return previous_position
        elif os_name == "Windows":
            return previous_position
        else:
            raise NotImplementedError(f"Unsupported system: {os_name}")
