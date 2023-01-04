"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import data_changed
import gl_utils
import numpy as np
import OpenGL.GL as gl
import pyglui.cygl.utils as cygl_utils
from observable import Observable
from plugin import System_Plugin_Base
from pyglui import ui
from pyglui.pyfontstash import fontstash as fs

COLOR_LEGEND_WORLD = cygl_utils.RGBA(0.66, 0.86, 0.461, 1.0)
COLOR_LEGEND_EYE_RIGHT = cygl_utils.RGBA(0.9844, 0.5938, 0.4023, 1.0)
COLOR_LEGEND_EYE_LEFT = cygl_utils.RGBA(0.668, 0.6133, 0.9453, 1.0)
NUMBER_SAMPLES_TIMELINE = 4000


class System_Timelines(Observable, System_Plugin_Base):
    def __init__(self, g_pool, show_world_fps=True, show_eye_fps=True):
        super().__init__(g_pool)
        self.show_world_fps = show_world_fps
        self.show_eye_fps = show_eye_fps
        self.cache_fps_data()
        self.pupil_positions_listener = data_changed.Listener(
            "pupil_positions", g_pool.rec_dir, plugin=self
        )
        self.pupil_positions_listener.add_observer(
            "on_data_changed", self._on_pupil_positions_changed
        )

    def init_ui(self):
        self.glfont = fs.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_font("opensans")
        self.fps_timeline = ui.Timeline(
            "Recorded FPS", self.draw_fps, self.draw_fps_legend
        )
        self.fps_timeline.content_height *= 2
        self.g_pool.user_timelines.append(self.fps_timeline)

    def deinit_ui(self):
        self.g_pool.user_timelines.remove(self.fps_timeline)
        self.fps_timeline = None

    def cache_fps_data(self):
        fps_world = self.calculate_fps(self.g_pool.timestamps)
        fps_eye0 = self.calculate_fps(self.g_pool.pupil_positions[0, ...].timestamps)
        fps_eye1 = self.calculate_fps(self.g_pool.pupil_positions[1, ...].timestamps)

        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]
        self.cache = {
            "world": fps_world,
            "eye0": fps_eye0,
            "eye1": fps_eye1,
            "xlim": [t0, t1],
            "ylim": [0, 210],
        }

    def calculate_fps(self, timestamps):
        if len(timestamps) > 1:
            timestamps = np.unique(timestamps)
            fps = 1.0 / np.diff(timestamps)
            return tuple(zip(timestamps, fps))
        return ()

    def draw_fps(self, width, height, scale):
        with gl_utils.Coord_System(*self.cache["xlim"], *self.cache["ylim"]):
            if self.show_world_fps:
                cygl_utils.draw_points(
                    self.cache["world"], size=2 * scale, color=COLOR_LEGEND_WORLD
                )
            if self.show_eye_fps:
                cygl_utils.draw_points(
                    self.cache["eye0"], size=2 * scale, color=COLOR_LEGEND_EYE_RIGHT
                )
                cygl_utils.draw_points(
                    self.cache["eye1"], size=2 * scale, color=COLOR_LEGEND_EYE_LEFT
                )

    def draw_fps_legend(self, width, height, scale):
        self.glfont.push_state()
        self.glfont.set_align_string(v_align="right", h_align="top")
        self.glfont.set_size(15.0 * scale)
        self.glfont.draw_text(width, 0, self.fps_timeline.label)

        legend_height = 13.0 * scale
        pad = 10 * scale

        if self.show_world_fps:
            self.glfont.draw_text(width, legend_height, "world FPS")
            cygl_utils.draw_polyline(
                [
                    (pad, legend_height + pad * 2 / 3),
                    (width / 2, legend_height + pad * 2 / 3),
                ],
                color=COLOR_LEGEND_WORLD,
                line_type=gl.GL_LINES,
                thickness=4.0 * scale,
            )
            legend_height += 1.5 * pad

        if self.show_eye_fps:
            self.glfont.draw_text(width, legend_height, "eye1 FPS")
            cygl_utils.draw_polyline(
                [
                    (pad, legend_height + pad * 2 / 3),
                    (width / 2, legend_height + pad * 2 / 3),
                ],
                color=COLOR_LEGEND_EYE_LEFT,
                line_type=gl.GL_LINES,
                thickness=4.0 * scale,
            )
            legend_height += 1.5 * pad

            self.glfont.draw_text(width, legend_height, "eye0 FPS")
            cygl_utils.draw_polyline(
                [
                    (pad, legend_height + pad * 2 / 3),
                    (width / 2, legend_height + pad * 2 / 3),
                ],
                color=COLOR_LEGEND_EYE_RIGHT,
                line_type=gl.GL_LINES,
                thickness=4.0 * scale,
            )

        self.glfont.pop_state()

    def _on_pupil_positions_changed(self):
        self.cache_fps_data()
        self.fps_timeline.refresh()
