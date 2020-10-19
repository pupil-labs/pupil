"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
import glob

import player_methods as pm
from plugin import Plugin
from observable import Observable

from video_overlay.workers.overlay_renderer import EyeOverlayRenderer
from video_overlay.models.config import Configuration
from video_overlay.ui.management import UIManagementEyes
from video_overlay.utils.constraints import ConstraintedValue, BooleanConstraint

from pupil_recording import PupilRecording


class Eye_Overlay(Observable, Plugin):
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

        self.current_frame_ts = None
        self.show_ellipses = ConstraintedValue(show_ellipses, BooleanConstraint())
        self._scale = scale
        self._alpha = alpha

        self.eye0 = self._setup_eye(0, eye0_config)
        self.eye1 = self._setup_eye(1, eye1_config)

    def recent_events(self, events):
        if "frame" in events:
            frame = events["frame"]
            self.current_frame_ts = frame.timestamp
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
        overlay = EyeOverlayRenderer(
            config, self.show_ellipses, self.make_current_pupil_datum_getter(eye_id)
        )
        return overlay

    def _video_path_for_eye(self, eye_id: int) -> str:
        # Get all eye videos for eye_id
        recording = PupilRecording(self.g_pool.rec_dir)
        eye_videos = list(recording.files().videos().eye_id(eye_id))

        if eye_videos:
            return str(eye_videos[0])
        else:
            return "/not/found/eye{}.mp4".format(eye_id)

    def get_init_dict(self):
        return {
            "scale": self.scale,
            "alpha": self.alpha,
            "show_ellipses": self.show_ellipses.value,
            "eye0_config": self.eye0.config.as_dict(),
            "eye1_config": self.eye1.config.as_dict(),
        }

    def make_current_pupil_datum_getter(self, eye_id):
        def _pupil_getter():
            try:
                pupil_data = self.g_pool.pupil_positions[eye_id, "2d"]
                if pupil_data:
                    closest_pupil_idx = pm.find_closest(
                        pupil_data.data_ts, self.current_frame_ts
                    )
                    current_datum_2d = pupil_data.data[closest_pupil_idx]
                else:
                    current_datum_2d = None

                pupil_data = self.g_pool.pupil_positions[eye_id, "3d"]

                if pupil_data:
                    closest_pupil_idx = pm.find_closest(
                        pupil_data.data_ts, self.current_frame_ts
                    )
                    current_datum_3d = pupil_data.data[closest_pupil_idx]
                else:
                    current_datum_3d = None
                return current_datum_2d, current_datum_3d
            except (IndexError, ValueError):
                return None

        return _pupil_getter
