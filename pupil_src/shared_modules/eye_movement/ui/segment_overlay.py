"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
import eye_movement.model as model
import player_methods as pm
import pyglui.cygl.utils as gl_utils
from pyglui.pyfontstash.fontstash import Context as GL_Font
import numpy as np
import cv2


def segment_draw(
    segment: model.Classified_Segment, size: t.Tuple[int, int], image=..., gl_font=...
):
    if image is not ...:
        segment_draw_on_image(segment=segment, size=size, image=image)
    else:
        gl_font = gl_font if gl_font is not ... else None
        segment_draw_in_gl_context(segment=segment, size=size, gl_font=gl_font)


def color_from_segment(segment: model.Classified_Segment) -> model.Color:
    return color_from_segment_class(segment.segment_class)


def color_from_segment_class(segment_class: model.Segment_Class) -> model.Color:
    return _DEFAULT_SEGMENT_CLASS_TO_COLOR_MAPPING[segment_class]


def _segment_class_to_color_mapping_with_palette(palette) -> t.Mapping[model.Segment_Class, t.Type[model.Color]]:
    return {
        model.Segment_Class.FIXATION: palette.yellow,
        model.Segment_Class.SACCADE: palette.green,
        model.Segment_Class.POST_SACCADIC_OSCILLATIONS: palette.blue,
        model.Segment_Class.SMOOTH_PURSUIT: palette.purple,
    }
_DEFAULT_SEGMENT_CLASS_TO_COLOR_MAPPING = _segment_class_to_color_mapping_with_palette(model.Defo_Color_Palette)


def segment_draw_on_image(
    segment: model.Classified_Segment, size: t.Tuple[int, int], image: np.ndarray
):
    if segment.segment_class == model.Segment_Class.SACCADE:
        _segment_draw_polyline_on_image(
            segment=segment,
            size=size,
            image=image,
        )
    elif segment.segment_class == model.Segment_Class.POST_SACCADIC_OSCILLATIONS:
        _segment_draw_polyline_on_image(
            segment=segment,
            size=size,
            image=image,
        )
    else:
        _segment_draw_mean_position_on_image(
            segment=segment,
            size=size,
            image=image,
        )


def _segment_draw_mean_position_on_image(
    segment: model.Classified_Segment, size: t.Tuple[int, int], image: np.ndarray
):
    segment_point = segment.mean_2d_point_within_world(size)
    segment_color = color_from_segment(segment).to_bgra().channels

    pm.transparent_circle(
        image, segment_point, radius=25.0, color=segment_color, thickness=3
    )

    _segment_draw_id_on_image(
        segment=segment,
        ref_point=segment_point,
        image=image,
    )

def _segment_draw_polyline_on_image(
    segment: model.Classified_Segment, size: t.Tuple[int, int], image: np.ndarray
):
    segment_points = segment.world_2d_points(size)
    polyline_color = color_from_segment(segment).to_bgr().channels
    polyline_thickness = 2

    if not segment_points:
        return

    cv2.polylines(
        image,
        np.asarray([segment_points], dtype=np.int32),
        isClosed=False,
        color=polyline_color,
        thickness=polyline_thickness,
        lineType=cv2.LINE_AA,
    )

    _segment_draw_id_on_image(
        segment=segment,
        ref_point=segment_points[-1],
        image=image,
    )


def _segment_draw_id_on_image(
    segment: model.Classified_Segment, ref_point: t.Tuple[int, int], image: np.ndarray
):
    text = str(segment.id)
    text_origin = (ref_point[0] + 30, ref_point[1])
    text_fg_color = color_from_segment(segment).to_bgr().channels
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    font_thickness = 1

    cv2.putText(
        img=image,
        text=text,
        org=text_origin,
        fontFace=font_face,
        fontScale=font_scale,
        color=text_fg_color,
        thickness=font_thickness,
    )


def segment_draw_in_gl_context(
    segment: model.Classified_Segment, size: t.Tuple[int, int], gl_font: GL_Font
):

    segment_point = segment.last_2d_point_within_world(size)
    circle_color = color_from_segment(segment).to_rgba().channels

    gl_utils.draw_circle(
        segment_point,
        radius=48.0,
        stroke_width=10.0,
        color=gl_utils.RGBA(*circle_color),
    )

    if gl_font:

        font_size = 22
        text = str(segment.id)
        text_origin_x = segment_point[0] + 48.0
        text_origin_y = segment_point[1]
        text_fg_color = color_from_segment(segment).to_rgba().channels

        gl_font.set_size(font_size)
        gl_font.set_color_float(text_fg_color)
        gl_font.draw_text(text_origin_x, text_origin_y, text)
