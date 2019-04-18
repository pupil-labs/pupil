import os
import glob

from plugin import Visualizer_Plugin_Base

from video_overlay.workers.overlay_renderer import OverlayRenderer
from video_overlay.models.config import Configuration


class Vis_Eye_Video_Overlay(Visualizer_Plugin_Base):
    icon_chr = chr(0xEC02)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, scale, alpha, eye0_config, eye1_config):
        super().__init__(g_pool)
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
        self.eye0.config.scale = val
        self.eye1.config.scale = val

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val
        self.eye0.config.alpha = val
        self.eye1.config.alpha = val

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
            "eye0_config": self.eye0.config.as_dict(),
            "eye1_config": self.eye1.config.as_dict(),
        }
