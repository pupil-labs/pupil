"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections

import gl_utils
import OpenGL.GL as gl
import pyglui.cygl.utils as cygl_utils
from pyglui import ui
from pyglui.pyfontstash import fontstash

Row = collections.namedtuple("Row", ["label", "elements"])


BarsElementTs = collections.namedtuple(
    "BarsElementTs", ["bar_positions_ts", "color_rgba", "width", "height"]
)
BarsElementTs.__new__.__defaults__ = ([], (1.0, 1.0, 1.0, 1.0), 3, 15)


RangeElementFramePerc = collections.namedtuple(
    "RangeElementFramePerc", ["from_perc", "to_perc", "color_rgba", "height", "offset"]
)
RangeElementFramePerc.__new__.__defaults__ = (0, 0, (1.0, 1.0, 1.0, 1.0), 8, 0)


RangeElementFrameIdx = collections.namedtuple(
    "RangeElementFrameIdx", ["from_idx", "to_idx", "color_rgba", "height", "offset"]
)
RangeElementFrameIdx.__new__.__defaults__ = (0, 0, (1.0, 1.0, 1.0, 1.0), 8, 0)


class PluginTimeline:
    timeline_row_height = 16

    def __init__(self, plugin, title, timeline_ui_parent, all_timestamps):
        self._timeline_ui_parent = timeline_ui_parent
        self._all_timestamps = all_timestamps
        self._time_start = all_timestamps[0]
        self._time_end = all_timestamps[-1]

        self._rows = []

        # initially set to minimum height
        self._timeline = ui.Timeline(
            title, self.draw_sections, self.draw_labels, content_height=1
        )

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_color_float((1.0, 1.0, 1.0, 1.0))
        self.glfont.set_align_string(v_align="right", h_align="top")

        plugin.add_observer("init_ui", self._on_init_ui)
        plugin.add_observer("deinit_ui", self._on_deinit_ui)

    def add_row(self, row):
        self._rows.append(row)

    def clear_rows(self):
        self._rows = []

    def refresh(self):
        self._timeline.content_height = max(
            0.001, self.timeline_row_height * len(self._rows)
        )
        self._timeline.refresh()

    def _on_init_ui(self):
        self._timeline_ui_parent.append(self._timeline)

    def _on_deinit_ui(self):
        self._timeline_ui_parent.remove(self._timeline)

    def draw_sections(self, width, height, scale):
        with gl_utils.Coord_System(
            left=self._time_start, right=self._time_end, bottom=height, top=0
        ):
            self._translate_to_vertical_center_of_row(scale)
            for row in self._rows:
                self._draw_row(row, height, scale)
                self._translate_to_next_row(scale)

    def _translate_to_vertical_center_of_row(self, scale):
        gl.glTranslatef(0, 0.001 + scale * self.timeline_row_height / 2, 0)

    def _draw_row(self, row, height, scale):
        for element in row.elements:
            if isinstance(element, BarsElementTs):
                self._draw_bars_element_ts(element, scale, height)
            elif isinstance(element, RangeElementFramePerc):
                self._draw_range_element_frame_perc(element, scale, height)
            elif isinstance(element, RangeElementFrameIdx):
                self._draw_range_element_frame_idx(element, scale, height)
            else:
                raise ValueError(f"Unknown element {element}")

    def _translate_to_next_row(self, scale):
        gl.glTranslatef(0, scale * self.timeline_row_height, 0)

    def _draw_bars_element_ts(self, element, scale, height):
        color = cygl_utils.RGBA(*element.color_rgba)
        cygl_utils.draw_bars(
            [(ts, 0) for ts in element.bar_positions_ts],
            height=element.height * scale,
            thickness=element.width * scale,
            color=color,
        )

    def _draw_range_element_frame_perc(self, element, scale, height):
        num_of_frames = len(self._all_timestamps)
        from_ts = self._all_timestamps[round(element.from_perc * num_of_frames)]
        to_ts = self._all_timestamps[round(element.to_perc * num_of_frames)]
        self._draw_range(
            from_ts, to_ts, scale, element.color_rgba, element.height, element.offset
        )

    def _draw_range_element_frame_idx(self, element, scale, height):
        self._draw_range(
            self._all_timestamps[element.from_idx],
            self._all_timestamps[element.to_idx],
            scale,
            element.color_rgba,
            element.height,
            element.offset,
        )

    def _draw_range(self, from_, to, scale, color_rgba, height, offset):
        gl.glTranslatef(0, offset * scale, 0)
        color = cygl_utils.RGBA(*color_rgba)
        cygl_utils.draw_rounded_rect(
            (from_, -height / 2 * scale),
            (to - from_, height * scale),
            corner_radius=0,
            color=color,
            sharpness=1.0,
        )
        gl.glTranslatef(0, -offset * scale, 0)

    def draw_labels(self, width, height, scale):
        self.glfont.set_size(self.timeline_row_height * scale)
        for row in self._rows:
            self.glfont.draw_text(width, 0, row.label)
            gl.glTranslatef(0, self.timeline_row_height * scale, 0)
