'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
import numpy as np

from pyglui import ui
from pyglui.pyfontstash import fontstash as fs
from pyglui.cygl.utils import *
import OpenGL.GL as gl

from plugin import System_Plugin_Base
import gl_utils

world_color = RGBA(0.66, 0.86, 0.461, 1.)
right_color = RGBA(0.9844, 0.5938, 0.4023, 1.)
left_color = RGBA(0.668, 0.6133, 0.9453, 1.)


class System_Timelines(System_Plugin_Base):
    def __init__(self, g_pool, show_world_fps=True, show_eye_fps=True):
        super().__init__(g_pool)
        self.show_world_fps = show_world_fps
        self.show_eye_fps = show_eye_fps
        self.cache_fps_data()

    def init_ui(self):
        self.glfont = fs.Context()
        self.glfont.add_font('opensans', ui.get_opensans_font_path())
        self.glfont.set_font('opensans')
        self.fps_timeline = ui.Timeline('Recorded FPS', self.draw_fps, self.draw_fps_legend)
        self.fps_timeline.content_height *= 2
        self.g_pool.user_timelines.append(self.fps_timeline)

    def deinit_ui(self):
        self.g_pool.user_timelines.remove(self.fps_timeline)
        self.fps_timeline = None

    def cache_fps_data(self):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]

        w_ts = np.asarray(self.g_pool.timestamps)
        w_fps = 1. / np.diff(w_ts)
        w_fps = [fps for fps in zip(w_ts, w_fps)]

        e0_ts = np.array([p['timestamp'] for p in self.g_pool.pupil_positions if p['id'] == 0])
        if e0_ts.shape[0] > 1:
            e0_fps = 1. / np.diff(e0_ts)
            e0_fps = [fps for fps in zip(e0_ts, e0_fps)]
        else:
            e0_fps = []

        e1_ts = np.array([p['timestamp'] for p in self.g_pool.pupil_positions if p['id'] == 1])
        if e1_ts.shape[0] > 1:
            e1_fps = 1. / np.diff(e1_ts)
            e1_fps = [fps for fps in zip(e1_ts, e1_fps)]
        else:
            e1_fps = []

        self.cache = {'world': w_fps, 'eye0': e0_fps, 'eye1': e1_fps,
                      'xlim': [t0, t1], 'ylim': [0, 210]}

    def draw_fps(self, width, height, scale):
        with gl_utils.Coord_System(*self.cache['xlim'], *self.cache['ylim']):
            if self.show_world_fps:
                draw_points(self.cache['world'], size=2*scale, color=world_color)
            if self.show_eye_fps:
                draw_points(self.cache['eye0'], size=2*scale, color=right_color)
                draw_points(self.cache['eye1'], size=2*scale, color=left_color)

    def draw_fps_legend(self, width, height, scale):
        self.glfont.push_state()
        self.glfont.set_align_string(v_align='right', h_align='top')
        self.glfont.set_size(15. * scale)
        self.glfont.draw_text(width, 0, self.fps_timeline.label)

        legend_height = 13. * scale
        pad = 10 * scale

        if self.show_world_fps:
            self.glfont.draw_text(width, legend_height, 'world FPS')
            draw_polyline([(pad, legend_height + pad * 2 / 3),
                           (width / 2, legend_height + pad * 2 / 3)],
                          color=world_color, line_type=gl.GL_LINES, thickness=4.*scale)
            legend_height += 1.5 * pad

        if self.show_eye_fps:
            self.glfont.draw_text(width, legend_height, 'eye1 FPS')
            draw_polyline([(pad, legend_height + pad * 2 / 3),
                           (width / 2, legend_height + pad * 2 / 3)],
                          color=left_color, line_type=gl.GL_LINES, thickness=4.*scale)
            legend_height += 1.5 * pad

            self.glfont.draw_text(width, legend_height, 'eye0 FPS')
            draw_polyline([(pad, legend_height + pad * 2 / 3),
                           (width / 2, legend_height + pad * 2 / 3)],
                          color=right_color, line_type=gl.GL_LINES, thickness=4.*scale)

        self.glfont.pop_state()

    def on_notify(self, notification):
        if notification['subject'] == 'pupil_positions_changed':
            self.cache_fps_data()
            self.fps_timeline.refresh()
