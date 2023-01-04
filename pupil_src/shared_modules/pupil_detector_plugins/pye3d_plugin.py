"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

import pye3d
from methods import normalize
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
from pyglui import ui
from version_utils import parse_version

from . import color_scheme
from .detector_base_plugin import PupilDetectorPlugin
from .visualizer_2d import draw_ellipse, draw_eyeball_outline, draw_pupil_outline
from .visualizer_pye3d import Eye_Visualizer

logger = logging.getLogger(__name__)

version_installed = parse_version(getattr(pye3d, "__version__", "0.0.1"))
version_supported = parse_version("0.3")

if not version_installed.release[:2] == version_installed.release[:2]:
    logger.info(
        f"Requires pye3d version {version_supported} "
        f"(Installed: {version_installed})"
    )
    raise ImportError("Unsupported version found")


class Pye3DPlugin(PupilDetectorPlugin):
    pupil_detection_identifier = "3d"
    # pupil_detection_method implemented as variable

    label = "Pye3D"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC19)
    order = 0.101

    @property
    def pupil_detector(self):
        return self.detector

    def __init__(self, g_pool=None, **kwargs):
        super().__init__(g_pool=g_pool)
        self.camera = CameraModel(
            focal_length=self.g_pool.capture.intrinsics.focal_length,
            resolution=self.g_pool.capture.intrinsics.resolution,
        )
        async_apps = ("capture", "service")
        mode = (
            DetectorMode.asynchronous
            if g_pool.app in async_apps
            else DetectorMode.blocking
        )
        logger.debug(f"Running {mode.name} in {g_pool.app}")
        self.detector = Detector3D(camera=self.camera, long_term_mode=mode, **kwargs)

        method_suffix = {
            DetectorMode.asynchronous: "real-time",
            DetectorMode.blocking: "post-hoc",
        }
        self.pupil_detection_method = f"pye3d {pye3d.__version__} {method_suffix[mode]}"

        self.debugVisualizer3D = Eye_Visualizer(self.g_pool, self.camera.focal_length)
        self.__debug_window_button = None

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        return init_dict

    def _process_camera_changes(self):
        camera = CameraModel(
            focal_length=self.g_pool.capture.intrinsics.focal_length,
            resolution=self.g_pool.capture.intrinsics.resolution,
        )
        if self.camera == camera:
            return

        logger.debug(f"Camera model change detected: {camera}. Resetting 3D detector.")
        self.camera = camera
        self.detector.reset_camera(self.camera)

        # Debug window also depends on focal_length, need to replace it with a new
        # instance. Make sure debug window is closed at this point or we leak the opengl
        # window.
        debug_window_was_opened = self.is_debug_window_open
        self.debug_window_close()
        self.debugVisualizer3D = Eye_Visualizer(self.g_pool, self.camera.focal_length)
        if debug_window_was_opened:
            self.debug_window_open()

    def on_resolution_change(self, old_size, new_size):
        # TODO: the logic for old 2D/3D resetting does not fit here anymore, but was
        # included in the PupilDetectorPlugin base class. This needs some cleaning up.
        pass

    def detect(self, frame, **kwargs):
        self._process_camera_changes()

        previous_detection_results = kwargs.get("previous_detection_results", [])
        for datum in previous_detection_results:
            if datum.get("method", "") == "2d c++":
                datum_2d = datum
                break
        else:
            logger.warning(
                "Required 2d pupil detection input not available. "
                "Returning default pye3d datum."
            )
            return self.create_pupil_datum(
                norm_pos=[0.5, 0.5],
                diameter=0.0,
                confidence=0.0,
                timestamp=frame.timestamp,
            )

        result = self.detector.update_and_detect(
            datum_2d, frame.gray, debug=self.is_debug_window_open
        )

        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )
        template = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )
        template.update(result)

        return template

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

        help_text = (
            f"pye3d {pye3d.__version__} - a model-based 3d pupil detector with corneal "
            "refraction correction. Read more about the detector in our docs website."
        )
        self.menu.append(ui.Info_Text(help_text))
        self.menu.append(ui.Button("Reset 3D model", self.reset_model))
        self.__debug_window_button = ui.Button(
            self.__debug_window_button_label, self.debug_window_toggle
        )

        help_text = (
            "The 3d model automatically updates in the background. Freeze the model to "
            "turn off automatic model updates. Refer to the docs website for details. "
        )
        self.menu.append(ui.Info_Text(help_text))
        self.menu.append(
            ui.Switch("is_long_term_model_frozen", self.detector, label="Freeze model")
        )
        self.menu.append(self.__debug_window_button)
        self.menu.append(ui.Info_Text("Color Legend - Default"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_3D.as_float, "3D pupil ellipse")
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_IN.as_float,
                "Long-term model outline (within bounds)",
            )
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_OUT.as_float,
                "Long-term model outline (out-of-bounds)",
            )
        )
        self.menu.append(ui.Info_Text("Color Legend - Debug"))
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_SHORT_TERM_DEBUG.as_float,
                "Short-term model outline",
            )
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_ULTRA_LONG_TERM_DEBUG.as_float,
                "Ultra-long-term model outline",
            )
        )

    def gl_display(self):
        self.debug_window_update()
        result = self._recent_detection_result

        if result is not None:

            # normal eyeball drawing
            draw_eyeball_outline(result)

            if self.is_debug_window_open and "debug_info" in result:
                # debug eyeball drawing
                debug_info = result["debug_info"]
                draw_ellipse(
                    ellipse=debug_info["projected_ultra_long_term"],
                    rgba=color_scheme.EYE_MODEL_OUTLINE_ULTRA_LONG_TERM_DEBUG.as_float,
                    thickness=2,
                )
                draw_ellipse(
                    ellipse=debug_info["projected_short_term"],
                    rgba=color_scheme.EYE_MODEL_OUTLINE_SHORT_TERM_DEBUG.as_float,
                    thickness=2,
                )

            # always draw pupil
            draw_pupil_outline(result, color_rgb=color_scheme.PUPIL_ELLIPSE_3D.as_float)

        if self.__debug_window_button:
            self.__debug_window_button.label = self.__debug_window_button_label

    def cleanup(self):
        # if we change detectors, be sure debug window is also closed
        self.debug_window_close()

    # Public

    def reset_model(self):
        self.detector.reset()

    # Debug window management

    @property
    def __debug_window_button_label(self) -> str:
        if not self.is_debug_window_open:
            return "Open debug window"
        else:
            return "Close debug window"

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
