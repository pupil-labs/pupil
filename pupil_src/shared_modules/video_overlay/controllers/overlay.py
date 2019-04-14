import logging
from collections import OrderedDict

import player_methods as pm
from observable import Observable

import video_overlay.utils.image_manipulation as IM
from video_overlay.controllers.config import Controller as ConfigController
from video_overlay.controllers.video import Controller as VideoController

logger = logging.getLogger(__name__)


class Controller(Observable):
    def __init__(self, video_path, config):
        self.attempt_to_load_video(video_path)
        self.config = ConfigController.from_updated_defaults(config)
        self.pipeline = self.setup_pipeline()

    def attempt_to_load_video(self, video_path):
        try:
            self.video = VideoController(video_path)
            self.valid_video_loaded = True
        except FileNotFoundError:
            logger.debug("Could not load overlay: {}".format(video_path))
            self.valid_video_loaded = False
        return self.valid_video_loaded

    def setup_pipeline(self):
        return OrderedDict(
            (
                (self.config.scale, IM.ScaleTransform()),
                (self.config.hflip, IM.HorizontalFlip()),
                (self.config.vflip, IM.VerticalFlip()),
            )
        )

    def draw_on_frame(self, target_frame):
        if not self.valid_video_loaded:
            return
        overlay_frame = self.video.closest_frame_to_ts(target_frame.timestamp)
        overlay_image = overlay_frame.img
        for param, manipulation in self.pipeline.items():
            overlay_image = manipulation.apply_to(overlay_image, param.value)
        self._render_overlay(target_frame.img, overlay_image)

    def _render_overlay(self, target_image, overlay_image):
        overlay_origin = (self.config.origin.x.value, self.config.origin.y.value)
        pm.transparent_image_overlay(
            overlay_origin, overlay_image, target_image, self.config.alpha.value
        )

    @property
    def video_path(self):
        return self.video.source.source_path if self.valid_video_loaded else None
