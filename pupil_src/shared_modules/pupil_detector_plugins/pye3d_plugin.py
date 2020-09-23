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

from pye3d.detector_3d import Detector3D
from pyglui import ui
from matplotlib import pyplot as plt
import pyqtgraph as pq

from .detector_base_plugin import PupilDetectorPlugin
from .visualizer_2d import draw_eyeball_outline, draw_pupil_outline
from .visualizer_pye3d import Eye_Visualizer

logger = logging.getLogger(__name__)


class Pye3DPlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC19)

    label = "Pye3D"
    identifier = "3d"
    order = 0.101

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.detector = Detector3D()
        self.debugVisualizer3D = Eye_Visualizer(
            g_pool, self.detector.settings["focal_length"]
        )

        self.data = []
        self.ts = []

    def detect(self, frame, **kwargs):
        previous_detection_results = kwargs.get("previous_detection_results", [])
        for datum in previous_detection_results:
            if datum.get("method", "") == "2d c++":
                datum_2d = datum
                break
        else:
            # TODO: make this more stable!
            raise RuntimeError("No 2D detection result! Needed for pye3D!")

        datum_2d["raw_edges"] = []
        result = self.detector.update_and_detect(datum_2d, frame.gray, debug=True)

        eye_id = self.g_pool.eye_id
        result["timestamp"] = frame.timestamp
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["id"] = eye_id
        result["method"] = "3d c++"

        if result["confidence"] > 0.6:
            hist = 400
            self.data.append(result["diameter_3d"])
            self.ts.append(frame.timestamp)
            self.data = self.data[-hist:]
            self.ts = self.ts[-hist:]

        global plotWidget
        try:
            plotWidget
        except NameError:
            plotWidget = pq.plot(title=f"Test {self.g_pool.eye_id}")

        plotWidget.clear()
        plotWidget.plot(self.ts, self.data)
        plotWidget.setYRange(0.5, 4.5)

        return result

    def on_notify(self, notification):
        super().on_notify(notification)

        subject = notification["subject"]
        if subject == "pupil_detector.3d.reset_model":
            if "id" not in notification:
                # simply apply to all eye processes
                self.reset_model()
            elif notification["id"] == self.g_pool.eye_id:
                # filter for specific eye processes
                self.reset_model()

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Pye3D Detector"

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name

        self.menu.append(ui.Button("Reset 3D model", self.reset_model))
        self.menu.append(ui.Button("Open debug window", self.debug_window_toggle))

        # self.menu.append(
        #     ui.Switch(TODO, label="Freeze model")
        # )

    def gl_display(self):
        self.debug_window_update()
        if self._recent_detection_result:
            draw_eyeball_outline(self._recent_detection_result)
            draw_pupil_outline(self._recent_detection_result)

    def cleanup(self):
        self.debug_window_close()  # if we change detectors, be sure debug window is also closed

    # Public

    def reset_model(self):
        self.detector.reset()

    # Debug window management

    @property
    def is_debug_window_open(self) -> bool:
        return self.debugVisualizer3D.window is not None

    def debug_window_toggle(self):
        if not self.is_debug_window_open:
            self.debug_window_open()
        else:
            self.debug_window_close()

    def debug_window_open(self):
        if not self.is_debug_window_open:
            self.debugVisualizer3D.open_window()

    def debug_window_close(self):
        if self.is_debug_window_open:
            self.debugVisualizer3D.close_window()

    def debug_window_update(self):
        if self.is_debug_window_open:
            self.debugVisualizer3D.update_window(
                self.g_pool, self._recent_detection_result
            )
