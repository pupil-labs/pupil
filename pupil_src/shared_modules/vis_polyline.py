"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from plugin import Visualizer_Plugin_Base
import numpy as np

import cv2

from pyglui import ui
from methods import denormalize

from scan_path.utils import np_denormalize


class Vis_Polyline(Visualizer_Plugin_Base):
    order = 0.9
    uniqueness = "not_unique"
    icon_chr = chr(0xE922)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, polyline_style_init_dict={}, **kwargs):
        super().__init__(g_pool)
        self.polyline_style_controller = PolylineStyleController(**polyline_style_init_dict)

    def get_init_dict(self):
        return {"polyline_style_init_dict": self.polyline_style_controller.get_init_dict()}

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return
        pts = self.previous_points(frame, events)
        if pts:
            pts = np.array([pts], dtype=np.int32)
            cv2.polylines(
                frame.img,
                pts,
                isClosed=False,
                color=self.polyline_style_controller.cv2_bgra,
                thickness=self.polyline_style_controller.thickness,
                lineType=cv2.LINE_AA,
            )

    def previous_points(self, frame, events):
        image_size = frame.img.shape[:-1][::-1]

        if self._scan_path_is_available(events):
            points_fields = ["norm_x", "norm_y"]
            gaze_data = events["scan_path_gaze"]
            gaze_points = gaze_data[points_fields]
            gaze_points = np.array(gaze_points.tolist(), dtype=gaze_points.dtype[0]) #FIXME: This is a workaround
            gaze_points = gaze_points.reshape((-1, len(points_fields)))
            gaze_points = np_denormalize(gaze_points, image_size, flip_y=True)
            return gaze_points.tolist()
        else:
            return [denormalize(datum["norm_pos"], image_size, flip_y=True) for datum in events.get("gaze", [])]

    def _scan_path_is_available(self, events):
        return events.get("scan_path_gaze", None) is not None

    def init_ui(self):

        polyline_style_thickness_slider = ui.Slider(
            "thickness",
            self.polyline_style_controller,
            min=self.polyline_style_controller.thickness_min,
            max=self.polyline_style_controller.thickness_max,
            step=self.polyline_style_controller.thickness_step,
            label="Line thickness",
        )

        polyline_style_color_info_text = ui.Info_Text("Set RGB color component values.")

        polyline_style_color_r_slider = ui.Slider(
            "r",
            self.polyline_style_controller,
            min=self.polyline_style_controller.rgba_min,
            max=self.polyline_style_controller.rgba_max,
            step=self.polyline_style_controller.rgba_step,
            label="Red"
        )
        polyline_style_color_g_slider = ui.Slider(
            "g",
            self.polyline_style_controller,
            min=self.polyline_style_controller.rgba_min,
            max=self.polyline_style_controller.rgba_max,
            step=self.polyline_style_controller.rgba_step,
            label="Green"
        )
        polyline_style_color_b_slider = ui.Slider(
            "b",
            self.polyline_style_controller,
            min=self.polyline_style_controller.rgba_min,
            max=self.polyline_style_controller.rgba_max,
            step=self.polyline_style_controller.rgba_step,
            label="Blue"
        )

        polyline_style_color_menu = ui.Growing_Menu("Color")
        polyline_style_color_menu.collapsed = True
        polyline_style_color_menu.append(polyline_style_color_info_text)
        polyline_style_color_menu.append(polyline_style_color_r_slider)
        polyline_style_color_menu.append(polyline_style_color_g_slider)
        polyline_style_color_menu.append(polyline_style_color_b_slider)

        self.menu.label = "Gaze Polyline"
        self.menu.append(polyline_style_thickness_slider)
        self.menu.append(polyline_style_color_menu)

    def deinit_ui(self):
        self.remove_menu()


class PolylineStyleController:

    rgba_min = 0.0
    rgba_max = 1.0
    rgba_step = 0.05

    thickness_min = 1
    thickness_max = 15
    thickness_step = 1

    def __init__(self, rgba=(1.0, 0.0, 0.4, 1.0), thickness=2):
        self.rgba = rgba
        self.thickness = thickness

    @property
    def rgba(self):
        return (self.r, self.g, self.b, self.a)

    @rgba.setter
    def rgba(self, rgba):
        self.r = rgba[0]
        self.g = rgba[1]
        self.b = rgba[2]
        self.a = rgba[3]

    def get_init_dict(self):
        return {"rgba": self.rgba, "thickness": self.thickness}

    @property
    def cv2_bgra(self):
        return (self.b*255, self.g*255, self.r*255, self.a*255)
