"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import cv2
import numpy as np
from data_changed import Listener
from methods import denormalize
from observable import Observable
from plugin import Plugin
from pyglui import ui
from scan_path import ScanPathController
from scan_path.utils import np_denormalize


class Vis_Polyline(Plugin, Observable):
    order = 0.9
    uniqueness = "not_unique"
    icon_chr = chr(0xE922)
    icon_font = "pupil_icons"

    def __init__(
        self, g_pool, polyline_style_init_dict={}, scan_path_init_dict={}, **kwargs
    ):
        super().__init__(g_pool)

        self.polyline_style_controller = PolylineStyleController(
            **polyline_style_init_dict
        )

        self.scan_path_controller = ScanPathController(g_pool, **scan_path_init_dict)
        self.scan_path_controller.add_observer(
            "on_update_ui", self._update_scan_path_ui
        )

        self._gaze_changed_listener = Listener(
            plugin=self, topic="gaze_positions", rec_dir=g_pool.rec_dir
        )
        self._gaze_changed_listener.add_observer(
            method_name="on_data_changed",
            observer=self.scan_path_controller.on_gaze_data_changed,
        )

    def get_init_dict(self):
        return {
            "polyline_style_init_dict": self.polyline_style_controller.get_init_dict(),
            "scan_path_init_dict": self.scan_path_controller.get_init_dict(),
        }

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
            label="Red",
        )
        polyline_style_color_g_slider = ui.Slider(
            "g",
            self.polyline_style_controller,
            min=self.polyline_style_controller.rgba_min,
            max=self.polyline_style_controller.rgba_max,
            step=self.polyline_style_controller.rgba_step,
            label="Green",
        )
        polyline_style_color_b_slider = ui.Slider(
            "b",
            self.polyline_style_controller,
            min=self.polyline_style_controller.rgba_min,
            max=self.polyline_style_controller.rgba_max,
            step=self.polyline_style_controller.rgba_step,
            label="Blue",
        )

        scan_path_timeframe_range = ui.Slider(
            "timeframe",
            self.scan_path_controller,
            min=self.scan_path_controller.min_timeframe,
            max=self.scan_path_controller.max_timeframe,
            step=self.scan_path_controller.timeframe_step,
            label="Duration",
        )

        scan_path_doc = ui.Info_Text("Duration of past gaze to include in polyline.")
        scan_path_status = ui.Info_Text("")

        polyline_style_color_menu = ui.Growing_Menu("Color")
        polyline_style_color_menu.collapsed = True
        polyline_style_color_menu.append(polyline_style_color_info_text)
        polyline_style_color_menu.append(polyline_style_color_r_slider)
        polyline_style_color_menu.append(polyline_style_color_g_slider)
        polyline_style_color_menu.append(polyline_style_color_b_slider)

        scan_path_menu = ui.Growing_Menu("Gaze History")
        scan_path_menu.collapsed = False
        scan_path_menu.append(scan_path_doc)
        scan_path_menu.append(scan_path_timeframe_range)
        scan_path_menu.append(scan_path_status)

        self.add_menu()
        self.menu.label = "Gaze Polyline"
        self.menu.append(polyline_style_thickness_slider)
        self.menu.append(polyline_style_color_menu)
        self.menu.append(scan_path_menu)

        self.scan_path_timeframe_range = scan_path_timeframe_range
        self.scan_path_status = scan_path_status

        self._update_scan_path_ui()

    def deinit_ui(self):
        self.remove_menu()
        self.scan_path_timeframe_range = None
        self.scan_path_status = None

    def recent_events(self, events):
        self.scan_path_controller.process()

        frame = events.get("frame")
        if not frame:
            return

        self._draw_polyline_path(frame, events)
        # self._draw_scan_path_debug(frame, events)

    def cleanup(self):
        self.scan_path_controller.cleanup()

    def _update_scan_path_ui(self):
        if self.menu_icon:
            self.menu_icon.indicator_stop = self.scan_path_controller.progress
        if self.scan_path_status:
            self.scan_path_status.text = self.scan_path_controller.status_string

    def _polyline_points(self, image_size, base_gaze_data, scan_path_gaze_data):
        if scan_path_gaze_data is not None:
            points_fields = ["norm_x", "norm_y"]
            gaze_points = scan_path_gaze_data[points_fields]
            gaze_points = np.array(
                gaze_points.tolist(), dtype=gaze_points.dtype[0]
            )  # FIXME: This is a workaround
            gaze_points = gaze_points.reshape((-1, len(points_fields)))
            gaze_points = np_denormalize(gaze_points, image_size, flip_y=True)
            return gaze_points.tolist()
        else:
            return [
                denormalize(datum["norm_pos"], image_size, flip_y=True)
                for datum in base_gaze_data
                if datum["confidence"] >= self.g_pool.min_data_confidence
            ]

    def _draw_polyline_path(self, frame, events):
        pts = self._polyline_points(
            image_size=frame.img.shape[:-1][::-1],
            base_gaze_data=events.get("gaze", []),
            scan_path_gaze_data=self.scan_path_controller.scan_path_gaze_for_frame(
                frame
            ),
        )

        if not pts:
            return

        pts = np.array([pts], dtype=np.int32)
        cv2.polylines(
            frame.img,
            pts,
            isClosed=False,
            color=self.polyline_style_controller.cv2_bgra,
            thickness=self.polyline_style_controller.thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_scan_path_debug(self, frame, events):
        from methods import denormalize
        from player_methods import transparent_circle

        gaze_data = self.scan_path_controller.scan_path_gaze_for_frame(frame)

        if gaze_data is None:
            return

        points_to_draw_count = len(gaze_data)
        image_size = frame.img.shape[:-1][::-1]

        for idx, datum in enumerate(gaze_data):
            point = (datum["norm_x"], datum["norm_y"])
            point = denormalize(point, image_size, flip_y=True)

            gray = float(idx) / points_to_draw_count
            transparent_circle(
                frame.img, point, radius=20, color=(gray, gray, gray, 0.9), thickness=2
            )


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
        return (self.b * 255, self.g * 255, self.r * 255, self.a * 255)
