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
from pupil_detectors import Detector2D, DetectorBase, Roi

from .detector_base_plugin import PupilDetectorPlugin


class Detector2DPlugin(PupilDetectorPlugin):
    uniqueness = "by_base_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "C++ 2d detector"
    identifier = "2d"

    def __init__(
        self, g_pool=None, namespaced_properties=None, detector_2d: Detector2D = None
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(namespaced_properties or {})

    @property
    def detector_properties_2d(self) -> dict:
        return self.detector_2d.get_properties()["2d"]

    def detect(self, frame):
        roi = Roi(*self.g_pool.u_r.get()[:4])
        result = self.detector_2d.detect(
            gray_img=frame.gray, color_img=frame.bgr, roi=roi
        )
        eye_id = self.g_pool.eye_id
        location = result["location"]
        result["norm_pos"] = normalize(
            location, (frame.width, frame.height), flip_y=True
        )
        result["timestamp"] = frame.timestamp
        result["topic"] = f"pupil.{eye_id}"
        result["id"] = eye_id
        result["method"] = "2d c++"
        return result

    def set_2d_detector_property(self, name, value):
        self.detector_2d.set_2d_detector_property(name, value)

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_2d

    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D"

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "
            + "Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "
            + "Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds."
        )
        self.menu.append(info)
        # self.menu.append(ui.Switch('coarse_detection',self.detector_properties_2d,label='Use coarse detection'))
        self.menu.append(
            ui.Slider(
                "intensity_range",
                self.detector_properties_2d,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_min",
                self.detector_properties_2d,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_max",
                self.detector_properties_2d,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
        # advanced_controls_menu = ui.Growing_Menu('Advanced Controls')
        # advanced_controls_menu.append(ui.Slider('contour_size_min',self.detector_properties_2d,label='Contour min length',min=1,max=200,step=1))
        # advanced_controls_menu.append(ui.Slider('ellipse_true_support_min_dist',self.detector_properties_2d,label='ellipse_true_support_min_dist',min=0.1,max=7,step=0.1))
        # self.menu.append(advanced_controls_menu)

    def deinit_ui(self):
        self.remove_menu()
