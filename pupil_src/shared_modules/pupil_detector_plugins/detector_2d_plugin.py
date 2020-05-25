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

from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

import glfw
from gl_utils import (
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from methods import normalize
from plugin import Plugin

from .detector_base_plugin import PropertyProxy, PupilDetectorPlugin
from .visualizer_2d import draw_pupil_outline

logger = logging.getLogger(__name__)


class Detector2DPlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "C++ 2d detector"
    identifier = "2d"
    order = 0.100

    def __init__(
        self, g_pool=None, namespaced_properties=None, detector_2d: Detector2D = None
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(namespaced_properties or {})
        self.proxy = PropertyProxy(self.detector_2d)

    def detect(self, frame, **kwargs):
        # convert roi-plugin to detector roi
        roi = Roi(*self.g_pool.roi.bounds)

        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None
        result = self.detector_2d.detect(
            gray_img=frame.gray, color_img=debug_img, roi=roi,
        )
        eye_id = self.g_pool.eye_id
        location = result["location"]
        result["norm_pos"] = normalize(
            location, (frame.width, frame.height), flip_y=True
        )
        result["timestamp"] = frame.timestamp
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["id"] = eye_id
        result["method"] = "2d c++"
        return result

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_2d

    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D"

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(0, 0.5, 1))

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "
            + "Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "
            + "Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Slider(
                "2d.intensity_range",
                self.proxy,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "2d.pupil_size_min",
                self.proxy,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "2d.pupil_size_max",
                self.proxy,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
