"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import typing as T

import glfw
from gl_utils import GLFWErrorReporting

GLFWErrorReporting.set_default()

try:
    from typing import OrderedDict as T_OrderedDict  # Python 3.7.2
except ImportError:

    class T_OrderedDict(collections.OrderedDict, T.MutableMapping[T.KT, T.VT]):
        pass


class GUIMonitor:
    """
    Wrapper class for monitor related GLFW API.
    """

    VideoMode = T.NamedTuple(
        "VideoMode",
        [
            ("width", int),
            ("height", int),
            ("red_bits", int),
            ("green_bits", int),
            ("blue_bits", int),
            ("refresh_rate", int),
        ],
    )

    __slots__ = ("__gl_handle", "__name", "__index")

    def __init__(self, index, gl_handle):
        self.__gl_handle = gl_handle
        self.__name = glfw.get_monitor_name(gl_handle).decode("utf-8")
        self.__index = index

    @property
    def unsafe_handle(self):
        return self.__gl_handle

    @property
    def name(self) -> str:
        tag = "PRIMARY" if self.__index == 0 else str(self.__index)
        return f"{self.__name} [{tag}]"

    @property
    def size(self) -> T.Tuple[int, int]:
        mode = self.current_video_mode
        return (mode.width, mode.height)

    @property
    def refresh_rate(self) -> int:
        mode = self.current_video_mode
        return mode.refresh_rate

    @property
    def current_video_mode(self) -> "GUIMonitor.VideoMode":
        gl_video_mode = glfw.get_video_mode(self.__gl_handle)
        return GUIMonitor.VideoMode(
            width=gl_video_mode.size.width,
            height=gl_video_mode.size.height,
            red_bits=gl_video_mode.bits.red,
            green_bits=gl_video_mode.bits.green,
            blue_bits=gl_video_mode.bits.blue,
            refresh_rate=gl_video_mode.refresh_rate,
        )

    @property
    def is_available(self) -> bool:
        return GUIMonitor.find_monitor_by_name(self.name) is not None

    @staticmethod
    def currently_connected_monitors() -> T.List["GUIMonitor"]:
        return [GUIMonitor(i, h) for i, h in enumerate(glfw.get_monitors())]

    @staticmethod
    def currently_connected_monitors_by_name() -> T_OrderedDict[str, "GUIMonitor"]:
        return collections.OrderedDict(
            (m.name, m) for m in GUIMonitor.currently_connected_monitors()
        )

    @staticmethod
    def primary_monitor() -> "GUIMonitor":
        gl_handle = glfw.get_primary_monitor()
        return GUIMonitor(0, gl_handle)

    @staticmethod
    def find_monitor_by_name(name: str) -> T.Optional["GUIMonitor"]:
        monitors = GUIMonitor.currently_connected_monitors_by_name()
        return monitors.get(name, None)
