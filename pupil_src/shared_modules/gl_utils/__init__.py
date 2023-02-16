"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .draw import draw_circle_filled_func_builder
from .trackball import Trackball
from .utils import (
    Coord_System,
    GLFWErrorReporting,
    _Rectangle,
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    current_context,
    cvmat_to_glmat,
    get_content_scale,
    get_framebuffer_scale,
    get_monitor_workarea_rect,
    get_window_frame_size_margins,
    glClear,
    glClearColor,
    glFlush,
    glViewport,
    is_window_visible,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
    window_coordinate_to_framebuffer_coordinate,
)
from .window_position_manager import WindowPositionManager
