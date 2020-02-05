import typing as T

from glfw import (
    glfwGetMonitors,
    glfwGetMonitorName,
    glfwGetPrimaryMonitor,
    glfwGetVideoMode,
)


class GUIMonitor:
    """
    Wrapper class for monitor related GLFW API.
    """

    VideoMode = T.NamedTuple("VideoMode", [
        ("width", int),
        ("height", int),
        ("red_bits", int),
        ("green_bits", int),
        ("blue_bits", int),
        ("refresh_rate", int),
    ])

    __slots__ = ("__gl_handle", "__name")

    def __init__(self, gl_handle):
        self.__gl_handle = gl_handle
        self.__name = glfwGetMonitorName(gl_handle)

    @property
    def unsafe_handle(self):
        return self.__gl_handle

    @property
    def name(self) -> str:
        return self.__name

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
        gl_video_mode = glfwGetVideoMode(self.__gl_handle)
        return GUIMonitor.VideoMode(*gl_video_mode)

    @property
    def is_available(self) -> bool:
        return GUIMonitor.find_monitor_by_name(self.name) is not None

    @staticmethod
    def currently_connected_monitors() -> T.List["GUIMonitor"]:
        return [GUIMonitor(h) for h in glfwGetMonitors()]

    @staticmethod
    def currently_connected_monitors_by_name() -> T.Mapping[str, "GUIMonitor"]:
        return {m.name: m for m in GUIMonitor.currently_connected_monitors()}

    @staticmethod
    def primary_monitor() -> "GUIMonitor":
        gl_handle = glfwGetPrimaryMonitor()
        return GUIMonitor(gl_handle)

    @staticmethod
    def find_monitor_by_name(name: str) -> T.Optional["GUIMonitor"]:
        monitors = GUIMonitor.currently_connected_monitors_by_name()
        return monitors.get(name, None)
