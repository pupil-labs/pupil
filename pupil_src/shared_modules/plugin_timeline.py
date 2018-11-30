"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections

import OpenGL.GL as gl
import pyglui.cygl.utils as cygl_utils
from pyglui import ui
from pyglui.pyfontstash import fontstash

import gl_utils

Line = collections.namedtuple("Line", ["label", "elements"])
BarsElementTs = collections.namedtuple(
    "BarsElementTs", ["bar_positions_ts", "color_rgba"]
)
RangeElementPerc = collections.namedtuple(
    "HorBarElementPerc", ["from_perc", "to_perc", "color_rgba"]
)


class PluginTimeline:
    timeline_line_height = 16

    def __init__(self, title, plugin, user_timelines, time_start, time_end):
        self._user_timelines = user_timelines
        self._time_start = time_start
        self._time_end = time_end

        self._lines = []

        # set to minimum height
        self._timeline = ui.Timeline(title, self.draw_sections, self.draw_labels, 1)

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_color_float((1.0, 1.0, 1.0, 1.0))
        self.glfont.set_align_string(v_align="right", h_align="top")

        plugin.add_observer("init_ui", self._on_init_ui)

    def add_line(self, line):
        self._lines.append(line)

    def clear_lines(self):
        self._lines = []

    def refresh(self):
        self._timeline.content_height = max(
            0.001, self.timeline_line_height * len(self._lines)
        )
        self._timeline.refresh()

    def _on_init_ui(self):
        self._user_timelines.append(self._timeline)

    def draw_sections(self, width, height, scale):
        for line_index, line in enumerate(self._lines):
            for element in line.elements:
                if isinstance(element, BarsElementTs):
                    self._draw_bars_element_ts(line_index, element, scale, height)
                elif isinstance(element, RangeElementPerc):
                    self._draw_range_element_perc(line_index, element, scale, height)
                else:
                    raise ValueError("Unknown element {}".format(element))

    def _draw_bars_element_ts(self, line_index, element, scale, height):
        with gl_utils.Coord_System(
            left=self._time_start, right=self._time_end, bottom=height, top=0
        ):
            gl.glTranslatef(0, scale * self.timeline_line_height * line_index, 0)
            gl.glTranslatef(0, 0.001 + scale * self.timeline_line_height / 2, 0)
            color = cygl_utils.RGBA(*element.color_rgba)
            cygl_utils.draw_bars(
                [(ts, 0) for ts in element.bar_positions_ts],
                height=15 * scale,
                thickness=3 * scale,
                color=color,
            )

    def _draw_range_element_perc(self, line_index, element, scale, height):
        with gl_utils.Coord_System(left=0, right=1, bottom=height, top=0):
            gl.glTranslatef(0, scale * self.timeline_line_height * line_index, 0)
            gl.glTranslatef(0, 0.001 + scale * self.timeline_line_height / 2, 0)
            color = cygl_utils.RGBA(*element.color_rgba)
            cygl_utils.draw_rounded_rect(
                (element.from_perc, -4 * scale),
                (element.to_perc, 8 * scale),
                corner_radius=0,
                color=color,
                sharpness=1.0,
            )

    def draw_labels(self, width, height, scale):
        self.glfont.set_size(self.timeline_line_height * scale)
        for line in self._lines:
            self.glfont.draw_text(width, 0, line.label)
            gl.glTranslatef(0, self.timeline_line_height * scale, 0)
