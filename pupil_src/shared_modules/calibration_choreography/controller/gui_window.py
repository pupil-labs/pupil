import collections
import platform
import typing as T
import contextlib

from gl_utils import adjust_gl_view
from gl_utils import basic_gl_setup
from glfw import *

from pyglui.cygl.utils import draw_polyline

from observable import Observable

from .gui_monitor import GUIMonitor


class GUIWindow(Observable):
    def __init__(self):
        self.__gl_handle = None

    @property
    def unsafe_handle(self):
        return self.__gl_handle

    @property
    def hdpi_factor(self) -> float:
        if self.__gl_handle is not None:
            return getHDPIFactor(self.__gl_handle)
        else:
            return 1.0

    @property
    def is_open(self) -> bool:
        return self.__gl_handle is not None

    def open(self, gui_monitor: GUIMonitor, title: str, is_fullscreen:bool=False, size:T.Tuple[int, int]=None, position:T.Tuple[int, int]=None):
        if self.is_open:
            # TODO: Warn that the window is already open
            return

        if not gui_monitor.is_available:
            raise ValueError(f"Window requires an available monitor.")

        has_fixed_size = (size is not None) and (len(size) == 2) and (size[0] > 0) and (size[1] > 0)
        if is_fullscreen and has_fixed_size:
            raise ValueError(f"Fullscreen is mutually exclusive to having a fixed size.")

        if position is None:
            if platform.system() == "Windows":
                position = (8, 90)
            else:
                position = (0, 0)

        if is_fullscreen:
            size = gui_monitor.size

        # NOTE: Always creating windowed window here, even if in fullscreen mode. On
        # windows you might experience a black screen for up to 1 sec when creating
        # a blank window directly in fullscreen mode. By creating it windowed and
        # then switching to fullscreen it will stay white the entire time.
        self.__gl_handle = glfwCreateWindow(*size, title, share=glfwGetCurrentContext())

        if not is_fullscreen:
            glfwSetWindowPos(self.__gl_handle, *position)

        # Register callbacks
        glfwSetFramebufferSizeCallback(self.__gl_handle, self.on_resize)
        glfwSetKeyCallback(self.__gl_handle, self.on_key)
        glfwSetMouseButtonCallback(self.__gl_handle, self.on_mouse_button)
        self.on_resize(self.__gl_handle, *glfwGetFramebufferSize(self.__gl_handle))

        # gl_state settings
        with self._switch_to_current_context():
            basic_gl_setup()
            glfwSwapInterval(0)

        if is_fullscreen:
            # Switch to full screen here. See NOTE above at glfwCreateWindow().
            glfwSetWindowMonitor(
                self.__gl_handle, gui_monitor.unsafe_handle, 0, 0, *gui_monitor.size, gui_monitor.refresh_rate
            )

    def close(self):
        if not self.is_open:
            return
        self.on_will_close()
        with self._switch_to_current_context():
            glfwSetInputMode(self.__gl_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL)
            glfwDestroyWindow(self.__gl_handle)
        self.__gl_handle = None
        self.on_did_close()

    def gl_display(self):
        if self.__gl_handle is None:
            return

        if glfwWindowShouldClose(self.__gl_handle):
            self.close()
            return

    @property
    def window_size(self) -> T.Tuple[int, int]:
        if self.__gl_handle is not None:
            return glfwGetFramebufferSize(self.__gl_handle)
        else:
            return (0, 0)

    def on_will_close(self):
        pass

    def on_did_close(self):
        pass

    def on_resize(self, gl_handle, w, h):
        with self._switch_to_current_context():
            adjust_gl_view(w, h)

    def on_key(self, gl_handle, key, scancode, action, mods):
        pass

    def on_mouse_button(self, gl_handle, button, action, mods):
        pass

    @contextlib.contextmanager
    def _switch_to_current_context(self):
        previous_context = glfwGetCurrentContext()
        glfwMakeContextCurrent(self.__gl_handle)
        try:
            yield
        finally:
            glfwMakeContextCurrent(previous_context)
