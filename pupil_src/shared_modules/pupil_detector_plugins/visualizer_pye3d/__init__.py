"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import math
from collections import deque

import numpy as np
from gl_utils.trackball import Trackball
from OpenGL.GL import (
    GL_LINES,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    glBegin,
    glColor4f,
    glEnd,
    glLineWidth,
    glLoadIdentity,
    glLoadMatrixf,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glScale,
    glTranslatef,
    glVertex3f,
)
from pyglui.cygl import utils as glutils
from pyglui.cygl.utils import RGBA
from visualizer import Visualizer

from .eye import LeGrandEye


class Eye_Visualizer(Visualizer):
    def __init__(self, g_pool, focal_length):
        super().__init__(g_pool, "Debug Visualizer", False)

        self.focal_length = focal_length
        self.image_width = 192  # right values are assigned in update
        self.image_height = 192

        camera_fov = math.degrees(
            2.0 * math.atan(self.image_height / (2.0 * self.focal_length))
        )
        self.trackball = Trackball(camera_fov)

        # self.residuals_single_means = deque(np.zeros(50), 50)
        # self.residuals_single_stds = deque(np.zeros(50), 50)
        # self.residuals_means = deque(np.zeros(50), 50)
        # self.residuals_stds = deque(np.zeros(50), 50)

        self.eye = LeGrandEye()
        self.cost_history = deque([], maxlen=200)
        self.optimization_number = 0

    ############## MATRIX FUNCTIONS ############

    def get_anthropomorphic_matrix(self):
        temp = np.identity(4)
        temp[2, 2] *= -1
        return temp

    def get_adjusted_pixel_space_matrix(self, scale):
        # returns a homoegenous matrix
        temp = self.get_anthropomorphic_matrix()
        temp[3, 3] *= scale
        return temp

    def get_image_space_matrix(self, scale=1.0):
        temp = self.get_adjusted_pixel_space_matrix(scale)
        temp[1, 1] *= -1  # image origin is top left
        temp[0, 3] = -self.image_width / 2.0
        temp[1, 3] = self.image_height / 2.0
        temp[2, 3] = -self.focal_length
        return temp.T

    ############## DRAWING FUNCTIONS ###########

    def draw_eye(self, result):

        self.eye.pupil_radius = result["circle_3d"]["radius"]
        self.eye.move_to_point(
            [
                result["sphere"]["center"][0],
                -result["sphere"]["center"][1],
                -result["sphere"]["center"][2],
            ]
        )
        if result["confidence"] > 0.0 and not np.isnan(
            result["circle_3d"]["normal"][0]
        ):
            self.eye.update_from_gaze_vector(
                [
                    result["circle_3d"]["normal"][0],
                    -result["circle_3d"]["normal"][1],
                    result["circle_3d"]["normal"][2],
                ]
            )
            self.eye.draw_gl(alpha=0.7)

    def draw_debug_info(self, result):

        sphere_center = result["sphere"]["center"]
        gaze_vector = result["circle_3d"]["normal"]
        pupil_radius = result["circle_3d"]["radius"]

        status = (
            "Eyeball center : X: %.2fmm Y: %.2fmm Z: %.2fmm \n"
            "Gaze vector:  X: %.2f Y: %.2f Z: %.2f\n"
            "Pupil Diameter: %.2fmm\n"
            "No. of supporting pupil observations: %i\n"
            % (
                sphere_center[0],
                sphere_center[1],
                sphere_center[2],
                gaze_vector[0],
                gaze_vector[1],
                gaze_vector[2],
                pupil_radius * 2,
                len(result["debug_info"]["Dierkes_lines"]),
            )
        )

        self.glfont.push_state()
        self.glfont.set_color_float((0, 0, 0, 1))
        self.glfont.draw_multi_line_text(7, 20, status)
        self.glfont.pop_state()

    def draw_residuals(self, result):

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)  # gl coord convention

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glTranslatef(0.01, 0.01, 0)

        glScale(1.5, 1.5, 1)

        glColor4f(1, 1, 1, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0.15, 0)
        glVertex3f(0.15, 0.15, 0)
        glVertex3f(0.15, 0, 0)
        glEnd()

        glTranslatef(0.01, 0.01, 0)

        glScale(0.13, 0.13, 1)

        vertices = [[0, 0], [0, 1]]
        glutils.draw_polyline(vertices, thickness=2, color=RGBA(1.0, 1.0, 1.0, 0.9))
        vertices = [[0, 0], [1, 0]]
        glutils.draw_polyline(vertices, thickness=2, color=RGBA(1.0, 1.0, 1.0, 0.9))

        glScale(1, 0.33, 1)

        vertices = [[0, 1], [1, 1]]
        glutils.draw_polyline(vertices, thickness=1, color=RGBA(1.0, 1.0, 1.0, 0.9))
        vertices = [[0, 2], [1, 2]]
        glutils.draw_polyline(vertices, thickness=1, color=RGBA(1.0, 1.0, 1.0, 0.9))
        vertices = [[0, 3], [1, 3]]
        glutils.draw_polyline(vertices, thickness=1, color=RGBA(1.0, 1.0, 1.0, 0.9))

        try:

            vertices = list(
                zip(
                    np.clip(
                        (np.asarray(result["debug_info"]["angles"]) - 10) / 40.0, 0, 1
                    ),
                    np.clip(
                        np.log10(np.array(result["debug_info"]["residuals"])) + 2,
                        0.1,
                        3.9,
                    ),
                )
            )

            alpha = 0.2 / (len(result["debug_info"]["angles"]) ** 0.2)
            size = 2 + 10 / (len(result["debug_info"]["angles"]) ** 0.1)

            glutils.draw_points(
                vertices, size=size, color=RGBA(255 / 255, 165 / 255, 0, alpha)
            )

        except:

            pass

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

    def draw_Dierkes_lines(self, result):

        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glColor4f(1.0, 0.0, 0.0, 0.1)
        glLineWidth(1.0)
        for line in result["debug_info"]["Dierkes_lines"][::4]:
            glBegin(GL_LINES)
            glVertex3f(line[0], line[1], line[2])
            glVertex3f(line[3], line[4], line[5])
            glEnd()
        glPopMatrix()

    def update_window(self, g_pool, result):

        if not result:
            return

        if not self.window:
            return

        self.begin_update_window()
        self.image_width, self.image_height = g_pool.capture.frame_size
        self.clear_gl_screen()
        self.trackball.push()

        glLoadMatrixf(self.get_image_space_matrix(15))

        g_pool.image_tex.draw(
            quad=(
                (0, self.image_height),
                (self.image_width, self.image_height),
                (self.image_width, 0),
                (0, 0),
            ),
            alpha=1.0,
        )

        glLoadMatrixf(self.get_adjusted_pixel_space_matrix(15))
        self.draw_frustum(self.image_width, self.image_height, self.focal_length)
        glLoadMatrixf(self.get_anthropomorphic_matrix())

        self.eye.pose = self.get_anthropomorphic_matrix()

        self.draw_coordinate_system(4)
        self.draw_eye(result)
        self.draw_Dierkes_lines(result)
        self.trackball.pop()

        self.draw_debug_info(result)
        self.draw_residuals(result)

        self.end_update_window()

        return True

    ############ WINDOW CALLBACKS ###############

    def on_resize(self, window, w, h):
        Visualizer.on_resize(self, window, w, h)
        self.trackball.set_window_size(w, h)

    def on_window_char(self, window, char):
        if char == ord("r"):
            self.trackball.distance = [0, 0, -0.1]
            self.trackball.pitch = 0
            self.trackball.roll = 0

    def on_scroll(self, window, x, y):
        self.trackball.zoom_to(y)
