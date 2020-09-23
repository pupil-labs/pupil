"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from .utils import *
from .trackball import *
from .window_position_manager import WindowPositionManager

# TODO: Remove calls to legacy functions
from .glfw_legacy import (
    legacy_glfw_get_video_mode,
    legacy_glfw_get_error,
    legacy_glfw_init,
    legacy_glfw_create_window,
    legacy_glfw_destroy_window,
    legacy_glfw_set_window_pos_callback,
    legacy_glfw_set_window_size_callback,
    legacy_glfw_set_window_close_callback,
    legacy_glfw_set_window_iconify_callback,
    legacy_glfw_set_framebuffer_size_callback,
    legacy_glfw_set_key_callback,
    legacy_glfw_set_char_callback,
    legacy_glfw_set_mouse_button_callback,
    legacy_glfw_set_cursor_pos_callback,
    legacy_glfw_set_scroll_callback,
    legacy_glfw_set_drop_callback
)
