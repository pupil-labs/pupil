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
import math

import numpy as np
import OpenGL
import OpenGL.error
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluErrorString

import glfw

# OpenGL.FULL_LOGGING = True
OpenGL.ERROR_LOGGING = False

logger = logging.getLogger(__name__)

__all__ = [
    "make_coord_system_norm_based",
    "make_coord_system_pixel_based",
    "make_coord_system_eye_camera_based",
    "adjust_gl_view",
    "clear_gl_screen",
    "basic_gl_setup",
    "cvmat_to_glmat",
    "is_window_visible",
    "Coord_System",
]


########################################################################################
# START OPENGL DEBUGGING

# We are injecting a custom error handling function into PyOpenGL to better investigate
# GLErrors that we are potentially producing in pyglui or cygl and that we don't catch
# appropriately. You can produce OpenGL errors with the following snippets for testing:

# import ctypes
# gl = ctypes.windll.LoadLibrary("opengl32") # NOTE: this is for windows, modify if needed

# # This produces `error 1281 (GL_INVALID_VALUE)`
# gl.glViewport(0, 0, -100, 1)

# # This produces `error 1280 (GL_INVALID_ENUM)` for some reason (!?)
# gl.glBegin()
# gl.glViewport(0, 0, -100, 1)
# gl.glEnd()

# # Check errors: (will consume flag)
# logger.debug(gl.glGetError())


_original_gl_error_check = OpenGL.error._ErrorChecker.glCheckError


def custom_gl_error_handling(
    error_checker, result, baseOperation=None, cArguments=None, *args
):
    try:
        return _original_gl_error_check(
            error_checker, result, baseOperation, cArguments, *args
        )
    except Exception as e:
        logging.error(f"Encountered PyOpenGL error: {e}")
        found_more_errors = False
        while True:
            err = glGetError()
            if err == GL_NO_ERROR:
                break
            if not found_more_errors:
                found_more_errors = True
                logging.debug("Emptying OpenGL error queue:")
            logging.debug(f"  glError: {err} -> {gluErrorString(err)}")
        if not found_more_errors:
            logging.debug("No more errors found in OpenGL error queue!")


OpenGL.error._ErrorChecker.glCheckError = custom_gl_error_handling

# END OPENGL DEBUGGING
########################################################################################


def is_window_visible(window):
    visible = glfw.glfwGetWindowAttrib(window, glfw.GLFW_VISIBLE)
    iconified = glfw.glfwGetWindowAttrib(window, glfw.GLFW_ICONIFIED)
    return visible and not iconified


def cvmat_to_glmat(m):
    mat = np.eye(4, dtype=np.float32)
    mat = mat.flatten()
    # convert to OpenGL matrix
    mat[0] = m[0, 0]
    mat[4] = m[0, 1]
    mat[12] = m[0, 2]
    mat[1] = m[1, 0]
    mat[5] = m[1, 1]
    mat[13] = m[1, 2]
    mat[3] = m[2, 0]
    mat[7] = m[2, 1]
    mat[15] = m[2, 2]
    return mat


def basic_gl_setup():
    glEnable(GL_POINT_SPRITE)
    # glEnable(GL_POINT_SMOOTH)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)  # overwrite pointsize
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glEnable(GL_LINE_SMOOTH)
    # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    # glEnable(GL_POLYGON_SMOOTH)
    # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)


def clear_gl_screen():
    glClear(GL_COLOR_BUFFER_BIT)


def adjust_gl_view(w, h):
    """
    adjust view onto our scene.
    """
    h = max(h, 1)
    w = max(w, 1)
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_pixel_based(img_shape, flip=False):
    height, width, channels = img_shape
    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(
            width, 0, 0, height, -1, 1
        )  # origin in the top left corner just like the img np-array
    else:
        glOrtho(
            0, width, height, 0, -1, 1
        )  # origin in the top left corner just like the img np-array

    # Switch back to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_eye_camera_based(window_size, focal_length):
    camera_fov = math.degrees(2.0 * math.atan(window_size[1] / (2.0 * focal_length)))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(camera_fov, float(window_size[0]) / window_size[1], 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_norm_based(flip=False):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(1, 0, 1, 0, -1, 1)  # gl coord convention
    else:
        glOrtho(0, 1, 0, 1, -1, 1)  # gl coord convention

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


class Coord_System(object):
    """docstring for Coord_System"""

    def __init__(self, left, right, bottom, top):
        super(Coord_System, self).__init__()
        if left == right:
            left -= 1
            right += 1
        if top == bottom:
            top -= 1
            bottom += 1
        self.bounds = left, right, bottom, top

    def __enter__(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(*self.bounds, -1, 1)  # gl coord convention

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def __exit__(self, *exc):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
