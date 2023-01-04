"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import contextlib
import logging
import math
import typing as T

import glfw
import numpy as np
import OpenGL.error
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_LINE_SMOOTH,
    GL_MODELVIEW,
    GL_NO_ERROR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SPRITE,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_VERTEX_PROGRAM_POINT_SIZE,
    glBlendFunc,
    glClear,
    glClearColor,
    glEnable,
    glFlush,
    glGetError,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glViewport,
)
from OpenGL.GLU import gluErrorString, gluPerspective

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
    "_Margins",
    "_Rectangle",
    "get_content_scale",
    "get_framebuffer_scale",
    "window_coordinate_to_framebuffer_coordinate",
    "get_monitor_workarea_rect",
    "get_window_content_rect",
    "get_window_frame_size_margins",
    "get_window_frame_rect",
    "get_window_title_bar_rect",
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
    visible = glfw.get_window_attrib(window, glfw.VISIBLE)
    iconified = glfw.get_window_attrib(window, glfw.ICONIFIED)
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


class Coord_System:
    """docstring for Coord_System"""

    def __init__(self, left, right, bottom, top):
        super().__init__()
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


class _Margins(T.NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


class _Rectangle(T.NamedTuple):
    x: int
    y: int
    width: int
    height: int

    def intersection(self, other: "_Rectangle") -> T.Optional["_Rectangle"]:
        in_min_x = max(self.x, other.x)
        in_min_y = max(self.y, other.y)

        in_max_x = min(self.x + self.width, other.x + other.width)
        in_max_y = min(self.y + self.height, other.y + other.height)

        if in_min_x < in_max_x and in_min_y < in_max_y:
            return _Rectangle(
                x=in_min_x,
                y=in_min_y,
                width=in_max_x - in_min_x,
                height=in_max_y - in_min_y,
            )
        else:
            return None


def get_content_scale(window) -> float:
    return glfw.get_window_content_scale(window)[0]


def get_framebuffer_scale(window) -> float:
    window_width = glfw.get_window_size(window)[0]
    framebuffer_width = glfw.get_framebuffer_size(window)[0]

    try:
        return float(framebuffer_width / window_width)
    except ZeroDivisionError:
        return 1.0


def window_coordinate_to_framebuffer_coordinate(window, x, y, cached_scale=None):
    scale = cached_scale or get_framebuffer_scale(window)
    return x * scale, y * scale


def get_monitor_workarea_rect(monitor) -> _Rectangle:
    x, y, w, h = glfw.get_monitor_workarea(monitor)
    return _Rectangle(x=x, y=y, width=w, height=h)


def get_window_content_rect(window) -> _Rectangle:
    x, y = glfw.get_window_pos(window)
    w, h = glfw.get_window_size(window)
    return _Rectangle(x=x, y=y, width=w, height=h)


def get_window_frame_size_margins(window) -> _Margins:
    left, top, right, bottom = glfw.get_window_frame_size(window)
    return _Margins(left=left, top=top, right=right, bottom=bottom)


def get_window_frame_rect(window) -> _Rectangle:
    content_rect = get_window_content_rect(window)
    frame_edges = get_window_frame_size_margins(window)
    return _Rectangle(
        x=content_rect.x - frame_edges.left,
        y=content_rect.y - frame_edges.top,
        width=content_rect.width + frame_edges.left + frame_edges.right,
        height=content_rect.height + frame_edges.top + frame_edges.bottom,
    )


def get_window_title_bar_rect(window) -> _Rectangle:
    frame_rect = get_window_frame_rect(window)
    frame_edges = get_window_frame_size_margins(window)
    return _Rectangle(
        x=frame_rect.x, y=frame_rect.y, width=frame_rect.width, height=frame_edges.top
    )


@contextlib.contextmanager
def current_context(window):
    prev_context = glfw.get_current_context()
    glfw.make_context_current(window)
    try:
        yield
    finally:
        glfw.make_context_current(prev_context)


_GLFWErrorReportingDict = T.Dict[T.Union[None, int], str]


class GLFWErrorReporting:
    @classmethod
    @contextlib.contextmanager
    def error_code_handling(
        cls,
        *_,
        ignore: T.Optional[T.Tuple[int]] = None,
        debug: T.Optional[T.Tuple[int]] = None,
        warn: T.Optional[T.Tuple[int]] = None,
        raise_: T.Optional[T.Tuple[int]] = None,
    ):
        old_reporting = glfw.ERROR_REPORTING

        if isinstance(old_reporting, dict):
            new_reporting: _GLFWErrorReportingDict = dict(old_reporting)
        else:
            new_reporting = cls.__default_error_reporting()

        new_reporting.update({err_code: "ignore" for err_code in ignore or ()})
        new_reporting.update({err_code: "log" for err_code in debug or ()})
        new_reporting.update({err_code: "warn" for err_code in warn or ()})
        new_reporting.update({err_code: "raise" for err_code in raise_ or ()})

        glfw.ERROR_REPORTING = new_reporting

        try:
            yield
        finally:
            glfw.ERROR_REPORTING = old_reporting

    @classmethod
    def set_default(cls):
        glfw.ERROR_REPORTING = cls.__default_error_reporting()

    ### Private

    @staticmethod
    def __default_error_reporting() -> _GLFWErrorReportingDict:
        ignore = [
            # GLFWError: (65544) b'Cocoa: Failed to find service port for display'
            # This happens on macOS Big Sur running on Apple Silicone hardware
            65544,
        ]
        return {
            None: "raise",
            **{code: "log" for code in ignore},
        }


GLFWErrorReporting.set_default()
