import collections
import platform
import typing as T
import contextlib

import OpenGL.GL as gl
import glfw
import glfw.GLFW  # TODO: Remove when switching to pyglfw API
import gl_utils

from pyglui.cygl.utils import draw_polyline

from observable import Observable

from .gui_monitor import GUIMonitor


class GUIWindow(Observable):
    def __init__(self):
        self.__gl_handle = None

    @property
    def unsafe_handle(self):
        """
        Raw pointer to the GL window context.
        """
        return self.__gl_handle

    @property
    def content_scale(self) -> float:
        if self.__gl_handle is not None:
            return gl_utils.get_content_scale(self.__gl_handle)
        else:
            return 1.0

    @property
    def window_size(self) -> T.Tuple[int, int]:
        if self.__gl_handle is not None:
            return glfw.GLFW.glfwGetFramebufferSize(self.__gl_handle)
        else:
            return (0, 0)

    @property
    def is_open(self) -> bool:
        return self.__gl_handle is not None

    def cursor_hide(self):
        glfw.GLFW.glfwSetInputMode(
            self.__gl_handle, glfw.GLFW.GLFW_CURSOR, glfw.GLFW.GLFW_CURSOR_HIDDEN
        )

    def cursor_disable(self):
        glfw.GLFW.glfwSetInputMode(
            self.__gl_handle, glfw.GLFW.GLFW_CURSOR, glfw.GLFW.GLFW_CURSOR_DISABLED
        )

    def open(
        self,
        gui_monitor: GUIMonitor,
        title: str,
        is_fullscreen: bool = False,
        size: T.Tuple[int, int] = None,
        position: T.Tuple[int, int] = None,
    ):
        if self.is_open:
            # TODO: Warn that the window is already open
            return

        if not gui_monitor.is_available:
            raise ValueError(f"Window requires an available monitor.")

        has_fixed_size = (
            (size is not None) and (len(size) == 2) and (size[0] > 0) and (size[1] > 0)
        )
        if is_fullscreen and has_fixed_size:
            raise ValueError(
                f"Fullscreen is mutually exclusive to having a fixed size."
            )

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
        self.__gl_handle = gl_utils.legacy_glfw_create_window(
            *size, title, share=glfw.GLFW.glfwGetCurrentContext()
        )

        if not is_fullscreen:
            glfw.GLFW.glfwSetWindowPos(self.__gl_handle, *position)

        # Register callbacks
        gl_utils.legacy_glfw_set_framebuffer_size_callback(
            self.__gl_handle, self.on_resize
        )
        gl_utils.legacy_glfw_set_key_callback(self.__gl_handle, self.on_key)
        gl_utils.legacy_glfw_set_mouse_button_callback(
            self.__gl_handle, self.on_mouse_button
        )
        self.on_resize(
            self.__gl_handle, *glfw.GLFW.glfwGetFramebufferSize(self.__gl_handle)
        )

        # gl_state settings
        with self._switch_to_current_context():
            gl_utils.basic_gl_setup()
            glfw.GLFW.glfwSwapInterval(0)

        if is_fullscreen:
            # Switch to full screen here. See NOTE above at legacy_glfw_create_window().
            glfw.GLFW.glfwSetWindowMonitor(
                self.__gl_handle,
                gui_monitor.unsafe_handle,
                0,
                0,
                *gui_monitor.size,
                gui_monitor.refresh_rate,
            )

    def close(self):
        if not self.is_open:
            return
        with self._switch_to_current_context():
            glfw.GLFW.glfwSetInputMode(
                self.__gl_handle, glfw.GLFW.GLFW_CURSOR, glfw.GLFW.GLFW_CURSOR_NORMAL
            )
            gl_utils.legacy_glfw_destroy_window(self.__gl_handle)
        self.__gl_handle = None

    @contextlib.contextmanager
    def drawing_context(self):
        if self.__gl_handle is None:
            return

        if glfw.GLFW.glfwWindowShouldClose(self.__gl_handle):
            self.close()
            return

        with self._switch_to_current_context():
            gl_utils.clear_gl_screen()

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, self.window_size[0], self.window_size[1], 0, -1, 1)
            # Switch back to Model View Matrix
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            yield self.unsafe_handle

            glfw.GLFW.glfwSwapBuffers(self.unsafe_handle)

    def on_resize(self, gl_handle, w, h):
        with self._switch_to_current_context():
            gl_utils.adjust_gl_view(w, h)

    def on_key(self, gl_handle, key, scancode, action, mods):
        if action == glfw.GLFW.GLFW_PRESS:
            if key == glfw.GLFW.GLFW_KEY_ESCAPE:
                self.on_key_press_escape()

    def on_mouse_button(self, gl_handle, button, action, mods):
        if action == glfw.GLFW.GLFW_PRESS:
            self.on_left_click()

    def on_left_click(self):
        pass

    def on_key_press_escape(self):
        pass

    @contextlib.contextmanager
    def _switch_to_current_context(self):
        previous_context = glfw.GLFW.glfwGetCurrentContext()
        glfw.GLFW.glfwMakeContextCurrent(self.__gl_handle)
        try:
            yield
        finally:
            glfw.GLFW.glfwMakeContextCurrent(previous_context)
