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
import typing as T
from enum import Enum

import numpy as np
from OpenGL.GL import GL_LINE_LOOP
from pyglui.cygl.utils import RGBA as cygl_rgba
from pyglui.cygl.utils import draw_points as cygl_draw_points
from pyglui.cygl.utils import draw_polyline as cygl_draw_polyline

import glfw
from methods import denormalize, normalize
from plugin import Plugin

logger = logging.getLogger(__name__)


# Type aliases
# Note that this version of Vec2 is immutable! We don't need mutability here.
Vec2 = T.Tuple[int, int]
Bounds = T.Tuple[int, int, int, int]


class RoiModel:
    """Model for ROI masks on an image frame.

    The mask has 2 primary properties:
        - frame_size: width, height
        - bounds: minx, miny, maxx, maxy

    Some notes on behavior:
    - Modifying bounds will always confine to the frame size.
    - Changing the frame size will scale the bounds to the same relative area.
    - If any frame dimension is <= 0, the ROI becomes invalid.
    - Setting the frame size of an invalid ROI to a valid size re-initializes the ROI.
    """

    def __init__(self, frame_size: Vec2) -> None:
        """Create a new RoiModel with bounds set to the full frame."""
        width, height = (int(v) for v in frame_size)
        self.frame_width = width
        self.frame_height = height
        self.minx = 0
        self.miny = 0
        self.maxx = width - 1
        self.maxy = height - 1

    def is_invalid(self) -> bool:
        """Returns true if the frame size has 0 dimension."""
        return self.frame_width <= 0 or self.frame_height <= 0

    def set_invalid(self) -> None:
        """Set frame size to (0, 0)."""
        self.frame_width = 0
        self.frame_height = 0

    @property
    def frame_size(self) -> Vec2:
        return self.frame_width, self.frame_height

    @frame_size.setter
    def frame_size(self, value: Vec2) -> None:
        """Set frame_size.

        Marks ROI as invalid, if size has 0 dimension.
        If old and new size are valid, scales the bounds to the same relative area.
        """
        width, height = (int(v) for v in value)
        if (width, height) == self.frame_size:
            return

        if width <= 0 or height <= 0:
            self.set_invalid()
            return

        # if we are recovering from invalid, just re-initialize
        if self.is_invalid():
            RoiModel.__init__(self, value)
            return

        # calculate scale factor for scaling bounds
        fx: float = width / self.frame_width
        fy: float = height / self.frame_height
        self.frame_width = width
        self.frame_height = height

        # scale bounds
        minx = int(round(self.minx * fx))
        miny = int(round(self.miny * fy))
        maxx = int(round(self.maxx * fx))
        maxy = int(round(self.maxy * fy))
        # set bounds (to also apply contrainsts)
        self.bounds = minx, miny, maxx, maxy

        logger.debug(f"Roi changed frame_size, now: {self}")

    @property
    def bounds(self) -> Bounds:
        return self.minx, self.miny, self.maxx, self.maxy

    @bounds.setter
    def bounds(self, value: Bounds) -> None:
        # convert to ints
        minx, miny, maxx, maxy = (int(v) for v in value)

        # ensure min < max, move max otherwise
        maxx = max(minx, maxx)
        maxy = max(miny, maxy)

        # ensure all 0 <= all bounds < dimension
        self.minx = min(max(minx, 0), self.frame_width - 1)
        self.miny = min(max(miny, 0), self.frame_height - 1)
        self.maxx = min(max(maxx, 0), self.frame_width - 1)
        self.maxy = min(max(maxy, 0), self.frame_height - 1)

    def __str__(self):
        return f"Roi(frame={self.frame_size}, bounds={self.bounds})"


class Handle(Enum):
    """Enum for the 4 handles of the ROI UI."""

    NONE = -1
    TOPLEFT = 0
    TOPRIGHT = 1
    BOTTOMRIGHT = 2
    BOTTOMLEFT = 3


class Roi(Plugin):
    """Plugin for managing a ROI on the frame."""

    # style definitions
    handle_size = 35
    handle_size_shadow = 45
    handle_size_active = 45
    handle_size_shadow_active = 65
    handle_color = cygl_rgba(0.5, 0.5, 0.9, 0.9)
    handle_color_active = cygl_rgba(0.5, 0.9, 0.9, 0.9)
    handle_color_shadow = cygl_rgba(0.0, 0.0, 0.0, 0.5)
    outline_color = cygl_rgba(0.8, 0, 0, 0.9)

    def __init__(
        self, g_pool, frame_size: Vec2 = (0, 0), bounds: Bounds = (0, 0, 0, 0),
    ) -> None:
        super().__init__(g_pool)
        self.model = RoiModel(frame_size)
        self.model.bounds = bounds

        self.active_handle = Handle.NONE
        self.reset_points()

        self.has_frame = False

    def reset_points(self) -> None:
        """Refresh cached points from underlying model."""
        # all points are in image coordinates
        # NOTE: for right/bottom points, we need to draw 1 behind the actual value. This
        # is because the outline is supposed to visually contain all pixels that are
        # masked.
        self._all_points = {
            Handle.TOPLEFT: (self.model.minx, self.model.miny),
            Handle.TOPRIGHT: (self.model.maxx + 1, self.model.miny),
            Handle.BOTTOMRIGHT: (self.model.maxx + 1, self.model.maxy + 1),
            Handle.BOTTOMLEFT: (self.model.minx, self.model.maxy + 1),
        }
        self._active_points = []
        self._inactive_points = []
        for handle, point in self._all_points.items():
            if handle == self.active_handle:
                self._active_points.append(point)
            else:
                self._inactive_points.append(point)

    def get_handle_at(self, pos: Vec2) -> Handle:
        """Returns which handle is rendered at that position."""
        for handle in self._all_points.keys():
            if self.is_point_on_handle(handle, pos):
                return handle
        return Handle.NONE

    def is_point_on_handle(self, handle: Handle, point: Vec2) -> bool:
        """Returns if point is within the rendered handle."""
        # NOTE: point and all stored points are in image coordinates. The render sizes
        # for the handles are in display coordinates! So we need to convert the points
        # in order for the distances to be correct.
        point_display = self.image_to_display_coordinates(point)
        center = self._all_points[handle]
        center_display = self.image_to_display_coordinates(center)
        distance = np.linalg.norm(
            (center_display[0] - point_display[0], center_display[1] - point_display[1])
        )
        handle_radius = self.g_pool.gui.scale * self.handle_size_shadow_active / 2
        return distance <= handle_radius

    def image_to_display_coordinates(self, point: Vec2) -> Vec2:
        """Convert image coordinates to display coordinates."""
        norm_pos = normalize(point, self.g_pool.capture.frame_size)
        return denormalize(norm_pos, self.g_pool.camera_render_size)

    # --- inherited from Plugin base class ---

    def recent_events(self, events: T.Dict[str, T.Any]) -> None:
        frame = events.get("frame")
        if frame is None:
            self.has_frame = False
            return

        self.has_frame = True
        self.model.frame_size = (frame.width, frame.height)
        self.reset_points()

    def on_click(self, pos: Vec2, button: int, action: int) -> bool:
        if action == glfw.GLFW_PRESS:
            clicked_handle = self.get_handle_at(pos)
            if clicked_handle != self.active_handle:
                self.active_handle = clicked_handle
                return True
        elif action == glfw.GLFW_RELEASE:
            if self.active_handle != Handle.NONE:
                self.active_handle = Handle.NONE
                return True
        return False

    def gl_display(self) -> None:
        if not self.has_frame or self.model.is_invalid():
            return

        # TODO: move down
        if self.g_pool.display_mode == "roi":
            return

        cygl_draw_polyline(
            self._all_points.values(),
            color=self.outline_color,
            thickness=1,
            line_type=GL_LINE_LOOP,
        )

        ui_scale = self.g_pool.gui.scale

        # draw inactive
        cygl_draw_points(
            self._inactive_points,
            size=ui_scale * self.handle_size_shadow,
            color=self.handle_color_shadow,
            sharpness=0.3,
        )
        cygl_draw_points(
            self._inactive_points,
            size=ui_scale * self.handle_size,
            color=self.handle_color,
            sharpness=0.9,
        )

        # draw active
        if self._active_points:
            cygl_draw_points(
                self._active_points,
                size=ui_scale * self.handle_size_shadow_active,
                color=self.handle_color_shadow,
                sharpness=0.3,
            )
            cygl_draw_points(
                self._active_points,
                size=ui_scale * self.handle_size_active,
                color=self.handle_color_active,
                sharpness=0.9,
            )

    def on_pos(self, pos: Vec2) -> None:
        if not self.has_frame or self.model.is_invalid():
            return

        if self.active_handle == Handle.NONE:
            return

        x, y = pos
        minx, miny, maxx, maxy = self.model.bounds

        min_size = 45
        if self.active_handle == Handle.TOPLEFT:
            minx = min(x, maxx - min_size)
            miny = min(y, maxy - min_size)
        elif self.active_handle == Handle.TOPRIGHT:
            maxx = max(minx + min_size, x)
            miny = min(y, maxy - min_size)
        elif self.active_handle == Handle.BOTTOMRIGHT:
            maxx = max(minx + min_size, x)
            maxy = max(miny + min_size, y)
        elif self.active_handle == Handle.BOTTOMLEFT:
            minx = min(x, maxx - min_size)
            maxy = max(miny + min_size, y)

        self.model.bounds = minx, miny, maxx, maxy
        self.reset_points()
