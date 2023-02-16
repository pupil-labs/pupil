"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import numpy as np
import OpenGL.GL as gl


def set_rotate_center(matrix):
    gl.glLoadTransposeMatrixf(matrix)


def shift_render_center(matrix):
    gl.glMultTransposeMatrixf(matrix)


def render_polygon_in_3d_window(vertices, color):
    r, g, b, a = color
    gl.glColor4f(r, g, b, 0.5)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    gl.glBegin(gl.GL_POLYGON)
    for vertex in vertices:
        gl.glVertex3f(*vertex)
    gl.glEnd()

    gl.glColor4f(*color)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    gl.glBegin(gl.GL_POLYGON)
    for vertex in vertices:
        gl.glVertex3f(*vertex)
    gl.glEnd()


def render_strip_in_3d_window(vertices, color):
    gl.glColor4f(*color)
    gl.glBegin(gl.GL_LINE_STRIP)
    for vertex in vertices:
        gl.glVertex3f(*vertex)
    gl.glEnd()


def render_centroid(color, point_size=5):
    gl.glLoadIdentity()
    gl.glPointSize(point_size)
    gl.glColor4f(*color)
    gl.glBegin(gl.GL_POINTS)
    gl.glVertex3f(0, 0, 0)
    gl.glEnd()


def render_coordinate(scale=1):
    origin = [0, 0, 0]
    colors = [(0.12, 0.46, 0.70, 1), (1.0, 0.49, 0.05, 1), (0.17, 0.62, 0.17, 1)]
    for axis, color in zip(np.eye(3), colors):
        render_strip_in_3d_window([origin, axis * scale], color)


def render_camera_frustum(camera_pose_matrix, camera_intrinsics, color):
    shift_render_center(camera_pose_matrix)
    render_coordinate()
    _render_frustum(camera_intrinsics.resolution, camera_intrinsics.K, color)


def _render_frustum(img_size, camera_matrix, color, scale=1500):
    x = img_size[0] / scale
    y = img_size[1] / scale
    z = (camera_matrix[0, 0] + camera_matrix[1, 1]) / scale

    origin = [0, 0, 0]
    vertices = []
    vertices += [origin, [x, y, z], [x, -y, z]]
    vertices += [origin, [x, y, z], [-x, y, z]]
    vertices += [origin, [-x, -y, z], [x, -y, z]]
    vertices += [origin, [-x, -y, z], [-x, y, z]]

    render_polygon_in_3d_window(vertices, color)


def render_camera_trace(recent_camera_trace, color):
    render_strip_in_3d_window(recent_camera_trace, color)
