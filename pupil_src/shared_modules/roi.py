"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T
from enum import Enum

import glfw
import numpy as np
from gl_utils import GLFWErrorReporting
from OpenGL.GL import GL_LINE_LOOP
from pyglui.cygl.utils import RGBA as cygl_rgba
from pyglui.cygl.utils import draw_points as cygl_draw_points
from pyglui.cygl.utils import draw_polyline as cygl_draw_polyline

GLFWErrorReporting.set_default()

from methods import denormalize, normalize
from observable import Observable
from plugin import Plugin

logger = logging.getLogger(__name__)


# Type aliases
# Note that this version of Vec2 is immutable! We don't need mutability here.
Vec2 = T.Tuple[int, int]
Bounds = T.Tuple[int, int, int, int]
ChangeCallback = T.Callable[[], None]


class RoiModel(Observable):
    """Model for ROI masks on an image frame.

    The mask has 2 primary properties:
        - frame_size: width, height
        - bounds: minx, miny, maxx, maxy

    Some notes on behavior:
    - Modifying bounds will always confine to the frame size and keep and area of >= 1.
    - Changing the frame size will scale the bounds to the same relative area.
    - If any frame dimension is <= 0, the ROI becomes invalid.
    - Setting the frame size of an invalid ROI to a valid size re-initializes the ROI.
    """

    def __init__(self, frame_size: Vec2) -> None:
        """Create a new RoiModel with bounds set to the full frame."""
        self._change_callbacks: T.List[ChangeCallback] = []
        self._set_to_full_frame(frame_size)

    def _set_to_full_frame(self, frame_size: Vec2) -> None:
        """Initialize to full frame for given frame_size."""
        width, height = (int(v) for v in frame_size)
        self._frame_width = width
        self._frame_height = height
        self._minx = 0
        self._miny = 0
        self._maxx = width - 1
        self._maxy = height - 1
        self.on_changed()

    def is_invalid(self) -> bool:
        """Returns true if the frame size has 0 dimension."""
        return self._frame_width <= 0 or self._frame_height <= 0

    def set_invalid(self) -> None:
        """Set frame size to (0, 0)."""
        self._frame_width = 0
        self._frame_height = 0
        self.on_changed()

    @property
    def frame_size(self) -> Vec2:
        return self._frame_width, self._frame_height

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
            self._set_to_full_frame(value)
            return

        # calculate scale factor for scaling bounds
        fx: float = width / self._frame_width
        fy: float = height / self._frame_height
        self._frame_width = width
        self._frame_height = height

        # scale bounds
        minx = int(round(self._minx * fx))
        miny = int(round(self._miny * fy))
        maxx = int(round(self._maxx * fx))
        maxy = int(round(self._maxy * fy))
        # set bounds (to also apply contrainsts)
        self.bounds = minx, miny, maxx, maxy

        self.on_changed()
        logger.debug(f"Roi changed frame_size, now: {self}")

    @property
    def bounds(self) -> Bounds:
        return self._minx, self._miny, self._maxx, self._maxy

    @bounds.setter
    def bounds(self, value: Bounds) -> None:
        # convert to ints
        minx, miny, maxx, maxy = (int(v) for v in value)

        # ensure all 0 <= all bounds < dimension
        minx = min(max(minx, 0), self._frame_width - 1)
        miny = min(max(miny, 0), self._frame_height - 1)
        maxx = min(max(maxx, 0), self._frame_width - 1)
        maxy = min(max(maxy, 0), self._frame_height - 1)

        # ensure min < max
        # tries move max behind min first, otherwise moves min before max
        if maxx <= minx:
            if minx < self._frame_width - 1:
                maxx = minx + 1
            else:
                minx = maxx - 1
        if maxy <= miny:
            if miny < self._frame_height - 1:
                maxy = miny + 1
            else:
                miny = maxy - 1

        self._minx, self._miny, self._maxx, self._maxy = minx, miny, maxx, maxy

        self.on_changed()

    def __str__(self):
        return f"Roi(frame={self.frame_size}, bounds={self.bounds})"

    def on_change(self, callback: ChangeCallback) -> None:
        """Register callback to be called when model changes."""
        self._change_callbacks.append(callback)

    def on_changed(self) -> None:
        """Called when the model changes.

        Observe this method to be notified of any changes.
        """
        pass


class Handle(Enum):
    """Enum for the 4 handles of the ROI UI."""

    NONE = -1
    TOPLEFT = 0
    TOPRIGHT = 1
    BOTTOMRIGHT = 2
    BOTTOMLEFT = 3


class Roi(Plugin):
    """Plugin for managing a ROI on the frame."""

    # Needs to be after base_backend and before detector_base_plugin!
    order = 0.05

    # style definitions
    handle_size = 35
    handle_size_shadow = 45
    handle_size_active = 45
    handle_size_shadow_active = 65
    handle_color = cygl_rgba(0.5, 0.5, 0.9, 0.9)
    handle_color_active = cygl_rgba(0.5, 0.9, 0.9, 0.9)
    handle_color_shadow = cygl_rgba(0.0, 0.0, 0.0, 0.5)
    outline_color = cygl_rgba(0.8, 0.8, 0.8, 0.9)

    def __init__(
        self,
        g_pool,
        frame_size: Vec2 = (0, 0),
        bounds: Bounds = (0, 0, 0, 0),
    ) -> None:
        super().__init__(g_pool)
        self.model = RoiModel(frame_size)
        self.model.bounds = bounds
        self._active_handle = Handle.NONE
        self.reset_points()
        self.model.add_observer("on_changed", self.reset_points)

        # Need to keep track of whether we have a valid frame to work with. Otherwise
        # don't render UI.
        self.has_frame = False

        # Expose roi model to outside.
        self.g_pool.roi = self.model

    def get_init_dict(self) -> T.Dict[str, T.Any]:
        return {
            "frame_size": self.model.frame_size,
            "bounds": self.model.bounds,
        }

    def reset_points(self) -> None:
        """Refresh cached points from underlying model."""
        if self.model.is_invalid():
            return
        # all points are in image coordinates
        # NOTE: for right/bottom points, we need to draw 1 behind the actual value. This
        # is because the outline is supposed to visually contain all pixels that are
        # masked.
        minx, miny, maxx, maxy = self.model.bounds
        self._all_points = {
            Handle.TOPLEFT: (minx, miny),
            Handle.TOPRIGHT: (maxx + 1, miny),
            Handle.BOTTOMRIGHT: (maxx + 1, maxy + 1),
            Handle.BOTTOMLEFT: (minx, maxy + 1),
        }
        self._active_points = []
        self._inactive_points = []
        for handle, point in self._all_points.items():
            if handle == self.active_handle:
                self._active_points.append(point)
            else:
                self._inactive_points.append(point)

    @property
    def active_handle(self) -> Handle:
        return self._active_handle

    @active_handle.setter
    def active_handle(self, value: Handle) -> None:
        """Set active handle. Will reset points when changed."""
        if value == self._active_handle:
            return
        self._active_handle = value
        self.reset_points()

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

    def on_click(self, pos: Vec2, button: int, action: int) -> bool:
        if not self.has_frame or self.model.is_invalid():
            return False

        if action == glfw.PRESS:
            clicked_handle = self.get_handle_at(pos)
            if clicked_handle != self.active_handle:
                self.active_handle = clicked_handle
                return True
        elif action == glfw.RELEASE:
            if self.active_handle != Handle.NONE:
                self.active_handle = Handle.NONE
                return True
        return False

    def gl_display(self) -> None:
        if not self.has_frame or self.model.is_invalid():
            return

        cygl_draw_polyline(
            self._all_points.values(),
            color=self.outline_color,
            thickness=1,
            line_type=GL_LINE_LOOP,
        )

        # only display rest of the UI when we're in ROI mode
        if self.g_pool.display_mode != "roi":
            return

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

        # Try to ensure that the roi has min_size in both dimensions. This is important
        # because otherwise the handles might overlap and the user cannot control the
        # ROI anymore. Keep in mind that we cannot assume that the ROI had min_size
        # before, since you can also modify from the network. This is purely a UI issue.
        # You can test this by e.g. setting the ROI over the network to (0, 0, 1, 1) or
        # to (190, 190, 191, 191) for the 192x192 image.
        # For every point we:
        #   1. Set corresponding coordinate pair
        #   2. Push other coordinate pair away to ensure min_size
        #   3. Other pair might have been pushed to frame bounds, if so push current
        #      pair back into the other direction.
        # If the frame_size is greater than min_size, we ensure a min_size ROI,
        # otherwise we ensure that ROI is the full frame_size.
        min_size = 45
        width, height = self.model.frame_size
        if self.active_handle == Handle.TOPLEFT:
            # 1.
            minx, miny = x, y
            # 2.
            maxx = max(maxx, min(minx + min_size, width - 1))
            maxy = max(maxy, min(miny + min_size, height - 1))
            # 3.
            minx = min(minx, max(maxx - min_size, 0))
            miny = min(miny, max(maxy - min_size, 0))
        elif self.active_handle == Handle.TOPRIGHT:
            # 1.
            maxx, miny = x, y
            # 2.
            minx = min(minx, max(maxx - min_size, 0))
            maxy = max(maxy, min(miny + min_size, height - 1))
            # 3.
            maxx = max(maxx, min(minx + min_size, width - 1))
            miny = min(miny, max(maxy - min_size, 0))
        elif self.active_handle == Handle.BOTTOMRIGHT:
            # 1.
            maxx, maxy = x, y
            # 2.
            minx = min(minx, max(maxx - min_size, 0))
            miny = min(miny, max(maxy - min_size, 0))
            # 3.
            maxx = max(maxx, min(minx + min_size, width - 1))
            maxy = max(maxy, min(miny + min_size, height - 1))
        elif self.active_handle == Handle.BOTTOMLEFT:
            # 1.
            minx, maxy = x, y
            # 2.
            maxx = max(maxx, min(minx + min_size, width - 1))
            miny = min(miny, max(maxy - min_size, 0))
            # 3.
            minx = min(minx, max(maxx - min_size, 0))
            maxy = max(maxy, min(miny + min_size, height - 1))

        self.model.bounds = minx, miny, maxx, maxy
