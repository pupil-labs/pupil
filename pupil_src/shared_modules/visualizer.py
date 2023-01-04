"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import gl_utils
import glfw
from gl_utils import GLFWErrorReporting

GLFWErrorReporting.set_default()

import math
from platform import system

import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_LINE_LOOP,
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_NICEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SPRITE,
    GL_POLYGON_SMOOTH,
    GL_POLYGON_SMOOTH_HINT,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_VERTEX_PROGRAM_POINT_SIZE,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glEnable,
    glEnd,
    glHint,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glScale,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from pyglui.cygl import utils as glutils
from pyglui.cygl.utils import RGBA
from pyglui.pyfontstash import fontstash as fs
from pyglui.ui import get_opensans_font_path

# UI Platform tweaks
if system() == "Linux":
    window_position_default = (0, 0)
elif system() == "Windows":
    window_position_default = (8, 90)
else:
    window_position_default = (0, 0)


class Visualizer:
    """docstring for Visualizer
    Visualizer is a base class for all visualizations in new windows
    """

    def __init__(self, g_pool, name="Visualizer", run_independently=False):

        self.name = name
        self.window_size = (640, 480)
        self.window = None
        self.input = None
        self.run_independently = run_independently
        self.sphere = None
        self.other_window = None
        self.g_pool = g_pool

    def begin_update_window(self):
        if self.window:
            if glfw.window_should_close(self.window):
                self.close_window()
                return

            self.other_window = glfw.get_current_context()
            glfw.make_context_current(self.window)

    def update_window(self):
        pass

    def end_update_window(self):
        if self.window:
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.make_context_current(self.other_window)

    ############## DRAWING FUNCTIONS ##############################

    def draw_frustum(self, width, height, length):

        W = width / 2.0
        H = height / 2.0
        Z = length
        # draw it
        glLineWidth(1)
        glColor4f(1, 0.5, 0, 0.5)
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 0, 0)
        glVertex3f(-W, H, Z)
        glVertex3f(W, H, Z)
        glVertex3f(0, 0, 0)
        glVertex3f(W, H, Z)
        glVertex3f(W, -H, Z)
        glVertex3f(0, 0, 0)
        glVertex3f(W, -H, Z)
        glVertex3f(-W, -H, Z)
        glVertex3f(0, 0, 0)
        glVertex3f(-W, -H, Z)
        glVertex3f(-W, H, Z)
        glEnd()

    def draw_coordinate_system(self, l=1):
        # Draw x-axis line. RED
        glLineWidth(2)
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(l, 0, 0)
        glEnd()

        # Draw y-axis line. GREEN.
        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, l, 0)
        glEnd()

        # Draw z-axis line. BLUE
        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, l)
        glEnd()

    def draw_sphere(
        self,
        sphere_position,
        sphere_radius,
        contours=45,
        color=RGBA(0.2, 0.5, 0.5, 0.5),
    ):

        glPushMatrix()
        glTranslatef(sphere_position[0], sphere_position[1], sphere_position[2])
        glScale(sphere_radius, sphere_radius, sphere_radius)
        self.sphere.draw(color, primitive_type=GL_LINE_STRIP)
        glPopMatrix()

    def basic_gl_setup(self):
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)  # overwrite pointsize
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glClearColor(0.8, 0.8, 0.8, 1.0)
        glEnable(GL_LINE_SMOOTH)
        # glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

    def adjust_gl_view(self, w, h):
        """
        adjust view onto our scene.
        """
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def clear_gl_screen(self):
        glClearColor(0.9, 0.9, 0.9, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

    def close_window(self):
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None

    def open_window(self):
        if not self.window:
            self.input = {"button": None, "mouse": (0, 0)}

            # get glfw started
            if self.run_independently:
                glfw.init()
                glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
                self.window = glfw.create_window(
                    self.window_size[0], self.window_size[1], self.name, None, None
                )
            else:
                self.window = glfw.create_window(
                    self.window_size[0],
                    self.window_size[1],
                    self.name,
                    None,
                    glfw.get_current_context(),
                )

            self.other_window = glfw.get_current_context()

            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            glfw.set_window_pos(
                self.window, window_position_default[0], window_position_default[1]
            )
            # Register callbacks window
            glfw.set_framebuffer_size_callback(self.window, self.on_resize)
            glfw.set_window_iconify_callback(self.window, self.on_iconify)
            glfw.set_key_callback(self.window, self.on_window_key)
            glfw.set_char_callback(self.window, self.on_window_char)
            glfw.set_mouse_button_callback(self.window, self.on_window_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self.on_pos)
            glfw.set_scroll_callback(self.window, self.on_scroll)

            # get glfw started
            if self.run_independently:
                glutils.init()
            self.basic_gl_setup()

            self.sphere = glutils.Sphere(20)

            self.glfont = fs.Context()
            self.glfont.add_font("opensans", get_opensans_font_path())
            self.glfont.set_size(18)
            self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))
            self.on_resize(self.window, *glfw.get_framebuffer_size(self.window))
            glfw.make_context_current(self.other_window)

    ############ window callbacks #################
    def on_resize(self, window, w, h):
        h = max(h, 1)
        w = max(w, 1)

        self.window_size = (w, h)
        active_window = glfw.get_current_context()
        glfw.make_context_current(window)
        self.adjust_gl_view(w, h)
        glfw.make_context_current(active_window)

    def on_window_mouse_button(self, window, button, action, mods):
        # self.gui.update_button(button,action,mods)
        if action == glfw.PRESS:
            self.input["button"] = button
            self.input["mouse"] = glfw.get_cursor_pos(window)
        if action == glfw.RELEASE:
            self.input["button"] = None

    def on_pos(self, window, x, y):
        x, y = gl_utils.window_coordinate_to_framebuffer_coordinate(
            window, x, y, cached_scale=None
        )
        # self.gui.update_mouse(x,y)
        if self.input["button"] == glfw.MOUSE_BUTTON_RIGHT:
            old_x, old_y = self.input["mouse"]
            self.trackball.drag_to(x - old_x, y - old_y)
            self.input["mouse"] = x, y
        if self.input["button"] == glfw.MOUSE_BUTTON_LEFT:
            old_x, old_y = self.input["mouse"]
            self.trackball.pan_to(x - old_x, y - old_y)
            self.input["mouse"] = x, y

    def on_window_char(self, window, char):
        pass

    def on_scroll(self, window, x, y):
        pass

    def on_iconify(self, window, iconified):
        pass

    def on_window_key(self, window, key, scancode, action, mods):
        pass
