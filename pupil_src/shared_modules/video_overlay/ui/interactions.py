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

from methods import denormalize, normalize


class Draggable:
    __slots__ = ("overlay", "drag_offset")

    def __init__(self, overlay):
        self.drag_offset = None
        self.overlay = overlay

    def on_click(self, pos, button, action):
        if not self.overlay.valid_video_loaded:
            return False  # click event has not been consumed

        click_engaged = action == glfw.PRESS
        if click_engaged and self._in_bounds(pos):
            self.drag_offset = self._calculate_offset(pos)
            return True
        self.drag_offset = None
        return False

    def on_pos(self, pos):
        if self.overlay.valid_video_loaded and self.drag_offset:
            self.overlay.config.origin.x.value = int(pos[0] + self.drag_offset[0])
            self.overlay.config.origin.y.value = int(pos[1] + self.drag_offset[1])

    def _in_bounds(self, pos):
        curr_x, curr_y = pos
        origin = self.overlay.config.origin
        width, height = self._effective_overlay_frame_size()
        x_in_bounds = origin.x.value < curr_x < origin.x.value + width
        y_in_bounds = origin.y.value < curr_y < origin.y.value + height
        return x_in_bounds and y_in_bounds

    def _calculate_offset(self, pos):
        curr_x, curr_y = pos
        origin = self.overlay.config.origin
        x_offset = origin.x.value - curr_x
        y_offset = origin.y.value - curr_y
        return (x_offset, y_offset)

    def _effective_overlay_frame_size(self):
        overlay_scale = self.overlay.config.scale.value
        overlay_width, overlay_height = self.overlay.video.source.frame_size
        overlay_width = round(overlay_width * overlay_scale)
        overlay_height = round(overlay_height * overlay_scale)
        return overlay_width, overlay_height


def current_mouse_pos(window, camera_render_size, frame_size):
    content_scale = gl_utils.get_content_scale(window)
    x, y = glfw.get_cursor_pos(glfw.get_current_context())
    pos = x * content_scale, y * content_scale
    pos = normalize(pos, camera_render_size)
    # Position in img pixels
    pos = denormalize(pos, frame_size)
    return (int(pos[0]), int(pos[1]))
