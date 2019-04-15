"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import os

import OpenGL.GL as gl

import gl_utils
import glfw

# FOR SMOOTH RENDERING
os.environ["GL_FSAA_MODE"] = "11"


class GLWindow(abc.ABC):
    def __init__(self, plugin):
        self._input = {"down": False, "mouse": (0, 0)}
        self._window = None
        self._trackball = self._init_trackball()

        self.open_visualization_window = True

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)
        plugin.add_observer("gl_display", self._on_gl_display)

    def switch_visualization_window(self, new_value):
        self.open_visualization_window = new_value
        if new_value:
            self._open()
        else:
            self._close()

    def _on_init_ui(self):
        self.switch_visualization_window(True)

    def _open(self):
        if self._window:
            return

        self._window = self._glfw_init(window_size=(1280, 1335), window_position=(0, 0))
        self._gl_state_settings(self._window)
        self._register_callbacks(self._window)

    @staticmethod
    def _init_trackball():
        trackball = gl_utils.Trackball()
        trackball.zoom_to(-100)
        return trackball

    @staticmethod
    def _glfw_init(window_size, window_position):
        glfw.glfwInit()
        window = glfw.glfwCreateWindow(
            *window_size,
            title="Head Poses Visualizer",
            share=glfw.glfwGetCurrentContext()
        )
        glfw.glfwSetWindowPos(window, *window_position)
        return window

    @staticmethod
    def _gl_state_settings(window):
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.basic_gl_setup()
        gl_utils.make_coord_system_norm_based()
        glfw.glfwSwapInterval(0)
        glfw.glfwMakeContextCurrent(active_window)

    def _register_callbacks(self, window):
        glfw.glfwSetFramebufferSizeCallback(window, self._on_resize_window)
        glfw.glfwSetMouseButtonCallback(window, self._on_window_mouse_button)
        glfw.glfwSetCursorPosCallback(window, self._on_window_pos)
        glfw.glfwSetScrollCallback(window, self._on_scroll)
        glfw.glfwSetWindowCloseCallback(window, self._on_close_window)

    def _on_resize_window(self, window, w, h):
        self._trackball.set_window_size(w, h)
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.glfwMakeContextCurrent(active_window)

    def _on_window_mouse_button(self, window, button, action, mods):
        if action == glfw.GLFW_PRESS:
            self._input["down"] = True
            self._input["mouse"] = glfw.glfwGetCursorPos(window)
        elif action == glfw.GLFW_RELEASE:
            self._input["down"] = False

    def _on_window_pos(self, window, x, y):
        if self._input["down"]:
            old_x, old_y = self._input["mouse"]
            self._trackball.drag_to(x - old_x, y - old_y)
            self._input["mouse"] = x, y

    def _on_scroll(self, window, x, y):
        self._trackball.zoom_to(y)

    def _on_close_window(self, window):
        self.switch_visualization_window(False)

    def _on_deinit_ui(self):
        self._close()

    def _close(self):
        if not self._window:
            return

        self._glfw_deinit(self._window)

    def _glfw_deinit(self, window):
        glfw.glfwDestroyWindow(window)
        self._window = None

    def _on_gl_display(self):
        if not self._window:
            return

        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(self._window)
        self._init_3d_window()
        self._trackball.push()

        self._render()

        self._trackball.pop()
        glfw.glfwSwapBuffers(self._window)
        glfw.glfwMakeContextCurrent(active_window)
        glfw.glfwPollEvents()

    @staticmethod
    def _init_3d_window():
        gl.glClearColor(0.9, 0.9, 0.9, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)

    @abc.abstractmethod
    def _render(self):
        pass
