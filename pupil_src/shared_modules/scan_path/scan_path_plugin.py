"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from observable import Observable
from plugin import Plugin
from pyglui import ui

from .scan_path_controller import ScanPathController


logger = logging.getLogger(__name__)


class ScanPathPlugin(Plugin, Observable):

    icon_chr = chr(0xE422)
    icon_font = "pupil_icons"
    order = 0.1

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Scan Path"

    def __init__(self, g_pool, scan_path_init_dict={}):
        super().__init__(g_pool)
        self._scan_path_controller = ScanPathController(g_pool, **scan_path_init_dict)
        self._scan_path_controller.add_observer(
            "on_update_ui", self._update_scan_path_ui
        )

    def get_init_dict(self):
        return {"scan_path_init_dict": self._scan_path_controller.get_init_dict()}

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Scan Path"

        self.scan_path_status = ui.Info_Text("")
        self.scan_path_timeframe_range = ui.Slider(
            "timeframe",
            self._scan_path_controller,
            min=self._scan_path_controller.min_timeframe,
            max=self._scan_path_controller.max_timeframe,
            step=self._scan_path_controller.timeframe_step,
            label="Timeframe",
        )

        self.menu.append(self.scan_path_timeframe_range)
        self.menu.append(self.scan_path_status)
        self._update_scan_path_ui()

    def deinit_ui(self):
        self.remove_menu()
        self.scan_path_status = None

    def recent_events(self, events):
        self._scan_path_controller.process()

        frame = events.get("frame", None)

        if not frame:
            return

        events["scan_path_gaze"] = self._scan_path_controller.scan_path_gaze_for_frame(
            frame
        )

        # self._debug_draw_scan_path(events)

    def _debug_draw_scan_path(self, events):
        from methods import denormalize
        from player_methods import transparent_circle

        frame = events["frame"]
        gaze_data = events["scan_path_gaze"]

        if gaze_data is None:
            return

        points_to_draw_count = len(gaze_data)
        image_size = frame.img.shape[:-1][::-1]

        for idx, datum in enumerate(gaze_data):
            point = (datum["norm_x"], datum["norm_y"])
            point = denormalize(point, image_size, flip_y=True)

            gray = float(idx) / points_to_draw_count
            transparent_circle(
                frame.img, point, radius=20, color=(gray, gray, gray, 0.9), thickness=2,
            )

    def on_notify(self, notification):
        self._scan_path_controller.on_notify(notification)

    def cleanup(self):
        self._scan_path_controller.cleanup()

    def _update_scan_path_ui(self):
        if self.menu_icon:
            self.menu_icon.indicator_stop = self._scan_path_controller.progress
        if self.scan_path_status:
            self.scan_path_status.text = self._scan_path_controller.status_string
