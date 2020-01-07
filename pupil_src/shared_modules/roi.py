import logging
import typing as T

import numpy as np
from OpenGL.GL import GL_LINE_LOOP
from pyglui.cygl.utils import RGBA as cygl_rgba
from pyglui.cygl.utils import draw_points as cygl_draw_points
from pyglui.cygl.utils import draw_polyline as cygl_draw_polyline

from plugin import Plugin

logger = logging.getLogger(__name__)


class Bounds:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def scale(self, fx, fy):
        self.min_x *= fx
        self.min_y *= fy
        self.max_x *= fx
        self.max_y *= fy


class Roi(Plugin):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._frame_size: T.Optional[np.ndarray] = None
        self._bounds: T.Optional[Bounds] = None
        self.active_points_idx = None

        self.handle_size = 35
        self.handle_size_shadow = 45
        self.handle_size_active = 45
        self.handle_size_shadow_active = 65
        self.outline_color = cygl_rgba(0.8, 0, 0, 0.9)
        self.handle_color = cygl_rgba(0.5, 0.5, 0.9, 0.9)
        self.handle_color_active = cygl_rgba(0.5, 0.9, 0.9, 0.9)
        self.handle_color_shadow = cygl_rgba(0.0, 0.0, 0.0, 0.5)

    @property
    def frame_size(self) -> np.ndarray:
        return self._frame_size

    @frame_size.setter
    def frame_size(self, value: T.Optional[T.Sequence[int]]):
        if value is None:
            if self._frame_size is None:
                return
            logger.debug("Setting frame_size to None, disabling Roi.")
            self._frame_size = None
            self._bounds = None
            return

        value = np.array(value)

        if self._frame_size is None:
            logger.debug("Enabling Roi.")
            self._frame_size = value
            width, height = value
            self._bounds = Bounds(1, 1, width - 1, heigh - 1)
        else:
            scale_factor = value / self._frame_size
            self._frame_size = value
            self._bounds.scale(*scale_factor)

    @property
    def bounds(self):
        return self._bounds

    def get_points(self):
        return [
            [self.bounds.min_x, self.bounds.min_y],
            [self.bounds.max_x, self.bounds.min_y],
            [self.bounds.max_x, self.bounds.max_y],
            [self.bounds.min_x, self.bounds.max_y],
        ]

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            self.frame_size = None
            return

        self.frame_size = (frame.width, frame.height)

    def gl_display(self):

        points = self.get_points()

        cygl_draw_polyline(
            points, color=self.outline_color, thickness=2, line_type=GL_LINE_LOOP,
        )

        # if self.g_pool.display_mode != "roi":
        #     return

        ui_scale = self.g_pool.gui.scale

        # split into active/inactive
        inactive = points
        active = []
        if self.active_points_idx is not None:
            active.append(inactive.pop(self.active_points_idx))

        # draw inactive
        cygl_draw_points(
            inactive,
            size=ui_scale * self.handle_size_shadow,
            color=self.handle_color_shadow,
            sharpness=0.3,
        )
        cygl_draw_points(
            inactive,
            size=ui_scale * self.handle_size,
            color=self.handle_color,
            sharpness=0.3,
        )

        # draw active
        if active:
            cygl_draw_points(
                active,
                size=ui_scale * self.handle_size_shadow_active,
                color=self.handle_color_shadow,
                sharpness=0.3,
            )
            cygl_draw_points(
                active,
                size=ui_scale * self.handle_size_active,
                color=self.handle_color_active,
                sharpness=0.3,
            )

    def on_click(self, pos, button, action):
        print("onclick", pos, button, action)

    def on_pos(self, pos):
        print("onpos", pos)
