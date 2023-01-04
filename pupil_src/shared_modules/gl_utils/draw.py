"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import functools
import typing as T

import numpy as np
import OpenGL.GL as gl
from pyglui.cygl.utils import RGBA


def draw_circle_filled_func_builder(cache_size: int = 0):
    _circle_points_around_zero_f = _circle_points_around_zero
    _circle_points_offset_f = _circle_points_offset

    if cache_size > 0:
        _circle_points_around_zero_f = functools.lru_cache(cache_size)(
            _circle_points_around_zero_f
        )
        _circle_points_offset_f = functools.lru_cache(cache_size)(
            _circle_points_offset_f
        )

    return functools.partial(
        _draw_circle_filled,
        _circle_points_around_zero_f=_circle_points_around_zero_f,
        _circle_points_offset_f=_circle_points_offset_f,
    )


def _draw_circle_filled(
    screen_point: T.Tuple[float, float],
    size: float,
    color: RGBA,
    num_points: int = 50,
    *,
    _circle_points_around_zero_f,
    _circle_points_offset_f,
):
    points = _circle_points_offset_f(
        screen_point,
        radius=size,
        num_points=num_points,
        flat=False,
        _circle_points_around_zero_f=_circle_points_around_zero_f,
    )
    gl.glColor4f(color.r, color.g, color.b, color.a)
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glVertexPointer(2, gl.GL_DOUBLE, 0, points)
    gl.glDrawArrays(gl.GL_POLYGON, 0, points.shape[0])


def _circle_points_offset(
    offset: T.Tuple[float, float],
    radius: float,
    num_points: int,
    flat: bool = True,
    *,
    _circle_points_around_zero_f,
) -> np.ndarray:
    # NOTE: .copy() to avoid modifying the cached result
    points = _circle_points_around_zero_f(radius, num_points).copy()
    points[:, 0] += offset[0]
    points[:, 1] += offset[1]
    if flat:
        points.shape = -1
    return points


def _circle_points_around_zero(radius: float, num_points: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, num_points, dtype=np.float64)
    t.shape = -1, 1
    points = np.hstack([np.cos(t), np.sin(t)])
    points *= radius
    return points
