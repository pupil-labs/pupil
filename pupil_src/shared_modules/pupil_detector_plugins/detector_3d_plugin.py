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
from distutils.version import LooseVersion as VersionFormat

import pupil_detectors
from pupil_detectors import Detector3D, DetectorBase, Roi
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

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
from .visualizer_2d import draw_eyeball_outline, draw_pupil_outline
from .visualizer_3d import Eye_Visualizer

logger = logging.getLogger(__name__)

if VersionFormat(pupil_detectors.__version__) < VersionFormat("1.0.5"):
    msg = (
        f"This version of Pupil requires pupil_detectors >= 1.0.5."
        f" You are running with pupil_detectors == {pupil_detectors.__version__}."
        f" Please upgrade to a newer version!"
    )
    logger.error(msg)
    raise RuntimeError(msg)


class Detector3DPlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC19)

    label = "C++ 3d detector"
    identifier = "3d"
    order = 0.101

    def __init__(
        self, g_pool=None, namespaced_properties=None, detector_3d: Detector3D = None
    ):
        super().__init__(g_pool=g_pool)
        self.detector_3d = detector_3d or Detector3D(namespaced_properties or {})
        self.proxy = PropertyProxy(self.detector_3d)
        # debug window
        self.debugVisualizer3D = Eye_Visualizer(g_pool, self.detector_3d.focal_length())

    def detect(self, frame, **kwargs):
        # convert roi-plugin to detector roi
        roi = Roi(*self.g_pool.roi.bounds)

        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None
        result = self.detector_3d.detect(
            gray_img=frame.gray,
            timestamp=frame.timestamp,
            color_img=debug_img,
            roi=roi,
            debug=self.is_debug_window_open,
            internal_raw_2d_data=kwargs.get("internal_raw_2d_data", None),
        )

        eye_id = self.g_pool.eye_id
        location = result["location"]
        result["norm_pos"] = normalize(
            location, (frame.width, frame.height), flip_y=True
        )
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["id"] = eye_id
        result["method"] = "3d c++"
        return result

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_3d

    ### PupilDetectorPlugin API

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Pupil Detector 3D"

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu.append(
            ui.Info_Text(
                "Open the debug window to see a visualization of the 3D pupil detection."
            )
        )
        self.menu.append(ui.Button("Reset 3D model", self.reset_model))
        self.menu.append(ui.Button("Open debug window", self.debug_window_toggle))
        model_sensitivity_slider = ui.Slider(
            "3d.model_sensitivity",
            self.proxy,
            label="Model sensitivity",
            min=0.990,
            max=1.0,
            step=0.0001,
        )
        model_sensitivity_slider.display_format = "%0.4f"
        self.menu.append(model_sensitivity_slider)
        self.menu.append(
            ui.Switch("3d.model_is_frozen", self.proxy, label="Freeze model")
        )

    def gl_display(self):
        self.debug_window_update()
        if self._recent_detection_result:
            draw_eyeball_outline(self._recent_detection_result)
            draw_pupil_outline(self._recent_detection_result)

    def cleanup(self):
        self.debug_window_close()  # if we change detectors, be sure debug window is also closed

    # Public

    def reset_model(self):
        self.detector_3d.reset_model()

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
                self.g_pool, self.detector_3d.debug_result
            )
