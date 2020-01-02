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


class Detector3DPlugin(PupilDetectorPlugin):
    uniqueness = "by_base_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC19)

    label = "C++ 3d detector"
    identifier = "3d"

    def __init__(
        self, g_pool=None, namespaced_properties=None, detector_3d: Detector3D = None
    ):
        super().__init__(g_pool=g_pool)
        self.detector_3d = detector_3d or Detector3D(namespaced_properties or {})
        self.proxy = PropertyProxy(self.detector_3d)
        # debug window
        self.debugVisualizer3D = Eye_Visualizer(g_pool, self.detector_3d.focal_length())

    def detect(self, frame):
        roi = Roi(*self.g_pool.u_r.get()[:4])
        if (
            not 0 <= roi.x_min <= roi.x_max < frame.width
            or not 0 <= roi.y_min <= roi.y_max < frame.height
        ):
            # TODO: Invalid ROIs can occur when switching camera resolutions, because we
            # adjust the roi only after all plugin recent_events() have been called.
            # Optimally we make a plugin out of the ROI and call its recent_events()
            # immediately after the backend, before the detection.
            logger.debug(f"Invalid Roi {roi} for img {frame.width}x{frame.height}!")
            return None

        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None
        result = self.detector_3d.detect(
            gray_img=frame.gray,
            timestamp=frame.timestamp,
            color_img=debug_img,
            roi=roi,
            debug=self.is_debug_window_open,
        )

        eye_id = self.g_pool.eye_id
        location = result["location"]
        result["norm_pos"] = normalize(
            location, (frame.width, frame.height), flip_y=True
        )
        result["topic"] = f"pupil.{eye_id}"
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
        self.add_menu()
        self.menu.label = self.pretty_class_name
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
        info_3d = ui.Info_Text(
            "Open the debug window to see a visualization of the 3D pupil detection."
        )
        self.menu.append(info_3d)
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

    def deinit_ui(self):
        self.remove_menu()

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
