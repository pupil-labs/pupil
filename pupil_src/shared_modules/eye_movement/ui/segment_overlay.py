"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import typing as t
import eye_movement.model as model
import player_methods as pm
import pyglui.cygl.utils as gl_utils
from pyglui.pyfontstash.fontstash import Context as GL_Font
import numpy as np
import cv2


def color_from_segment(segment: model.Classified_Segment) -> model.Color:
    """
    Segment color legend:
        - Yellow - Fixation
        - Green - Saccade
        - Blue - Post-saccadic oscillation
        - Purple - Smooth pursuit
    """
    return color_from_segment_class(segment.segment_class)


def color_from_segment_class(segment_class: model.Segment_Class) -> model.Color:
    return _DEFAULT_SEGMENT_CLASS_TO_COLOR_MAPPING[segment_class]


def _segment_class_to_color_mapping_with_palette(
    palette
) -> t.Mapping[model.Segment_Class, t.Type[model.Color]]:
    # Note: Keep this mapping in sync with doc of color_from_segment
    return {
        model.Segment_Class.FIXATION: palette.yellow,
        model.Segment_Class.SACCADE: palette.green,
        model.Segment_Class.POST_SACCADIC_OSCILLATIONS: palette.blue,
        model.Segment_Class.SMOOTH_PURSUIT: palette.purple,
    }


_DEFAULT_SEGMENT_CLASS_TO_COLOR_MAPPING = _segment_class_to_color_mapping_with_palette(
    model.Defo_Color_Palette
)


class Segment_Overlay_Renderer(abc.ABC):
    def draw(self, segment: model.Classified_Segment):
        self.draw_polyline(segment)

    @property
    @abc.abstractmethod
    def canvas_size(self) -> t.Tuple[int, int]:
        ...

    @abc.abstractmethod
    def draw_id(self, segment: model.Classified_Segment, ref_point: t.Tuple[int, int]):
        ...

    @abc.abstractmethod
    def draw_circle(self, segment: model.Classified_Segment):
        ...

    @abc.abstractmethod
    def draw_polyline(self, segment: model.Classified_Segment):
        ...


class Segment_Overlay_Image_Renderer(Segment_Overlay_Renderer):
    def __init__(self, canvas_size: t.Tuple[int, int], image: np.ndarray):
        self._image = image
        self._canvas_size = canvas_size

    @property
    def canvas_size(self) -> t.Tuple[int, int]:
        return self._canvas_size

    def draw_id(self, segment: model.Classified_Segment, ref_point: t.Tuple[int, int]):
        text = str(segment.id)
        text_origin = (ref_point[0] + 30, ref_point[1])
        text_fg_color = color_from_segment(segment).to_bgr().channels
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        font_thickness = 1

        cv2.putText(
            img=self._image,
            text=text,
            org=text_origin,
            fontFace=font_face,
            fontScale=font_scale,
            color=text_fg_color,
            thickness=font_thickness,
        )

    def draw_circle(self, segment: model.Classified_Segment):
        segment_point = segment.mean_2d_point_within_world(self.canvas_size)
        segment_color = color_from_segment(segment).to_bgra().channels

        pm.transparent_circle(
            self._image, segment_point, radius=25.0, color=segment_color, thickness=3
        )

        self.draw_id(segment=segment, ref_point=segment_point)

    def draw_polyline(self, segment: model.Classified_Segment):
        segment_points = segment.world_2d_points(self.canvas_size)
        polyline_color = color_from_segment(segment).to_bgr().channels
        polyline_thickness = 2

        if not segment_points:
            return

        cv2.polylines(
            self._image,
            np.asarray([segment_points], dtype=np.int32),
            isClosed=False,
            color=polyline_color,
            thickness=polyline_thickness,
            lineType=cv2.LINE_AA,
        )

        self.draw_id(segment=segment, ref_point=segment_points[-1])


class Segment_Overlay_GL_Context_Renderer(Segment_Overlay_Renderer):
    def __init__(self, canvas_size: t.Tuple[int, int], gl_font: t.Optional[GL_Font]):
        self._gl_font = gl_font
        self._canvas_size = canvas_size

    @property
    def canvas_size(self) -> t.Tuple[int, int]:
        return self._canvas_size

    def draw_id(self, segment: model.Classified_Segment, ref_point: t.Tuple[int, int]):
        if not self._gl_font:
            return

        font_size = 22
        text = str(segment.id)
        text_origin_x = ref_point[0] + 48.0
        text_origin_y = ref_point[1]
        text_fg_color = color_from_segment(segment).to_rgba().channels

        self._gl_font.set_size(font_size)
        self._gl_font.set_color_float(text_fg_color)
        self._gl_font.draw_text(text_origin_x, text_origin_y, text)

    def draw_circle(self, segment: model.Classified_Segment):
        segment_point = segment.last_2d_point_within_world(self._canvas_size)
        circle_color = color_from_segment(segment).to_rgba().channels

        gl_utils.draw_circle(
            segment_point,
            radius=48.0,
            stroke_width=10.0,
            color=gl_utils.RGBA(*circle_color),
        )

        self.draw_id(segment=segment, ref_point=segment_point)

    def draw_polyline(self, segment: model.Classified_Segment):
        segment_points = segment.world_2d_points(self._canvas_size)
        polyline_color = color_from_segment(segment).to_rgba().channels
        polyline_thickness = 2

        if not segment_points:
            return

        gl_utils.draw_polyline(
            verts=segment_points,
            thickness=float(polyline_thickness),
            color=gl_utils.RGBA(*polyline_color),
        )

        self.draw_id(segment=segment, ref_point=segment_points[-1])
