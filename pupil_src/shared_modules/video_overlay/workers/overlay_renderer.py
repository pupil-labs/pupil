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
from collections import OrderedDict

import player_methods as pm
import video_overlay.utils.image_manipulation as IM
from observable import Observable
from video_overlay.models.config import Configuration
from video_overlay.utils.constraints import InclusiveConstraint
from video_overlay.workers.frame_fetcher import FrameFetcher

logger = logging.getLogger(__name__)


class OverlayRenderer:
    def __init__(self, config):
        self.config = config
        self.attempt_to_load_video()
        self.pipeline = self.setup_pipeline()

    def attempt_to_load_video(self):
        try:
            self.video = FrameFetcher(self.config.video_path)
            self.valid_video_loaded = True
        except FileNotFoundError:
            logger.debug(f"Could not load overlay: {self.config.video_path}")
            self.valid_video_loaded = False
        return self.valid_video_loaded

    def setup_pipeline(self):
        return [
            (self.config.scale, IM.ScaleTransform()),
            (self.config.hflip, IM.HorizontalFlip()),
            (self.config.vflip, IM.VerticalFlip()),
        ]

    def draw_on_frame(self, target_frame):
        if not self.valid_video_loaded:
            return
        overlay_frame = self.video.closest_frame_to_ts(target_frame.timestamp)
        overlay_image = overlay_frame.img
        try:
            is_fake_frame = overlay_frame.is_fake
            # TODO: once we have the unified Frame class, we should just pass the frames
            # to the image manipulation pipeline (instead of the images). Then we can
            # get rid of the additional parameter stuff in ImageManipulation!
        except AttributeError:
            # TODO: this is only fore extra safety until we have a unified Frame class
            # and can be sure that everything ending up here has the 'is_fake'
            # attribute!
            logger.warning(
                f"Frame passed to overlay renderer does not have 'is_fake' attribute!"
                f" Frame: {overlay_frame}"
            )
            is_fake_frame = False
        for param, manipulation in self.pipeline:
            overlay_image = manipulation.apply_to(
                overlay_image, param.value, is_fake_frame=overlay_frame.is_fake
            )

        self._adjust_origin_constraint(target_frame.img, overlay_image)
        self._render_overlay(target_frame.img, overlay_image)

    def _adjust_origin_constraint(self, target_image, overlay_image):
        max_x = target_image.shape[1] - overlay_image.shape[1]
        max_y = target_image.shape[0] - overlay_image.shape[0]
        self.config.origin.x.constraint = InclusiveConstraint(low=0, high=max_x)
        self.config.origin.y.constraint = InclusiveConstraint(low=0, high=max_y)

    def _render_overlay(self, target_image, overlay_image):
        overlay_origin = (self.config.origin.x.value, self.config.origin.y.value)
        pm.transparent_image_overlay(
            overlay_origin, overlay_image, target_image, self.config.alpha.value
        )


class EyeOverlayRenderer(OverlayRenderer):
    def __init__(self, config, should_render_pupil_data, pupil_getter):
        super().__init__(config)
        pupil_renderer = (should_render_pupil_data, IM.PupilRenderer(pupil_getter))
        self.pipeline.insert(0, pupil_renderer)
