import os
import glob

from plugin import Plugin
from observable import Observable

from video_overlay.workers.overlay_renderer import OverlayRenderer
from video_overlay.models.config import Configuration
from video_overlay.ui.management import UIManagementEyes


class Vis_Eye_Video_Overlay(Observable, Plugin):
    icon_chr = chr(0xEC02)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        scale=0.6,
        alpha=0.8,
        show_ellipses=True,
        eye0_config=None,
        eye1_config=None,
    ):
        super().__init__(g_pool)
        eye0_config = eye0_config or {"vflip": True, "origin_x": 210, "origin_y": 60}
        eye1_config = eye1_config or {"hflip": True, "origin_x": 10, "origin_y": 60}

        self.show_ellipses = show_ellipses
        self._scale = scale
        self._alpha = alpha

        self.eye0 = self._setup_eye(0, eye0_config)
        self.eye1 = self._setup_eye(1, eye1_config)

    def recent_events(self, events):
        if "frame" in events:
            frame = events["frame"]
            for overlay in (self.eye0, self.eye1):
                overlay.draw_on_frame(frame)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = val
        self.eye0.config.scale.value = val
        self.eye1.config.scale.value = val

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val
        self.eye0.config.alpha.value = val
        self.eye1.config.alpha.value = val

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Eye Video Overlays"
        self.ui = UIManagementEyes(self, self.menu, (self.eye0, self.eye1))

    def deinit_ui(self):
        self.ui.teardown()
        self.remove_menu()

    def _setup_eye(self, eye_id, prefilled_config):
        video_path = self._video_path_for_eye(eye_id)
        prefilled_config["video_path"] = video_path
        prefilled_config["scale"] = self.scale
        prefilled_config["alpha"] = self.alpha
        config = Configuration(**prefilled_config)
        overlay = OverlayRenderer(config)
        return overlay

    def _video_path_for_eye(self, eye_id):
        rec_dir = self.g_pool.rec_dir
        video_file_pattern = "eye{}.*".format(eye_id)
        video_path_pattern = os.path.join(rec_dir, video_file_pattern)
        try:
            video_path_candidates = glob.iglob(video_path_pattern)
            return next(video_path_candidates)
        except StopIteration:
            return None

    def get_init_dict(self):
        return {
            "scale": self.scale,
            "alpha": self.alpha,
            "show_ellipses": self.show_ellipses,
            "eye0_config": self.eye0.config.as_dict(),
            "eye1_config": self.eye1.config.as_dict(),
        }
