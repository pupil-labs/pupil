"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

logger = logging.getLogger(__name__)

import pyglui
import pyglui.cygl.utils as pyglui_utils
import gl_utils


from surface_tracker import Heatmap_Mode
from surface_tracker.surface_tracker import Surface_Tracker
from surface_tracker.surface_online import Surface_Online


class Surface_Tracker_Online(Surface_Tracker):
    def __init__(self, g_pool, marker_min_perimeter=60, inverted_markers=False):
        self.Surface_Class = Surface_Online
        self.freeze_scene = False
        self.frozen_scene_frame = None
        self.frozen_scene_tex = None

        super().__init__(
            g_pool,
            marker_min_perimeter=marker_min_perimeter,
            inverted_markers=inverted_markers,
        )
        self.ui_info_text = "This plugin detects and tracks fiducial markers visible in the scene. You can define surfaces using 1 or more marker visible within the world view by clicking *add surface*. You can edit defined surfaces by selecting *Surface edit mode*."
        self.supported_heatmap_modes = [Heatmap_Mode.WITHIN_SURFACE]

        self.menu = None
        self.button = None
        self.add_button = None

    @property
    def save_dir(self):
        return self.g_pool.user_dir

    def _update_ui_custom(self):
        def set_freeze_scene(val):
            self.freeze_scene = val
            if val:
                self.frozen_scene_tex = pyglui_utils.Named_Texture()
                self.frozen_scene_tex.update_from_ndarray(self.current_frame.img)
            else:
                self.frozen_scene_tex = None

        self.menu.append(
            pyglui.ui.Switch(
                "freeze_scene", self, label="Freeze Scene", setter=set_freeze_scene
            )
        )

    def _per_surface_ui_custom(self, surface, s_menu):
        def set_gaze_hist_len(val):
            if val <= 0:
                logger.warning("Gaze history length must be a positive number!")
                return
            surface.gaze_history_length = val

        s_menu.append(
            pyglui.ui.Text_Input(
                "gaze_history_length",
                surface,
                label="Gaze History Length [seconds]",
                setter=set_gaze_hist_len,
            )
        )

    def recent_events(self, events):
        if self.freeze_scene:
            current_frame = events.get("frame")
            events["frame"] = self.current_frame

        super().recent_events(events)

        if not self.current_frame:
            return

        self._update_surface_gaze_history(events, self.current_frame.timestamp)

        if self.gui.show_heatmap:
            self._update_surface_heatmaps()

        if self.freeze_scene:
            events["frame"] = current_frame

    def _update_markers(self, frame):
        self._detect_markers(frame)

    def _update_surface_locations(self, idx):
        for surface in self.surfaces:
            surface.update_location(idx, self.markers, self.camera_model)

    def _update_surface_gaze_history(self, events, world_timestamp):
        surfaces_gaze_dict = {e["name"]: e["gaze_on_surf"] for e in events["surfaces"]}

        for surface in self.surfaces:
            try:
                surface.update_gaze_history(
                    surfaces_gaze_dict[surface.name], world_timestamp
                )
            except KeyError:
                pass

    def _update_surface_corners(self):
        for surface, corner_idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(
                    corner_idx, self._last_mouse_pos.copy(), self.camera_model
                )

    def _update_surface_heatmaps(self):
        for surface in self.surfaces:
            surface.update_heatmap(surface.gaze_history)

    def add_surface(self, _=None, init_dict=None):
        if self.freeze_scene:
            logger.warning("Surfaces cannot be added while the scene is frozen!")
        else:
            super().add_surface(init_dict=init_dict)

    def gl_display(self):
        if self.freeze_scene:
            self.gl_display_frozen_scene()
        super().gl_display()

    def gl_display_frozen_scene(self):
        gl_utils.clear_gl_screen()

        gl_utils.make_coord_system_norm_based()

        self.frozen_scene_tex.draw()

        gl_utils.make_coord_system_pixel_based(
            (self.g_pool.capture.frame_size[1], self.g_pool.capture.frame_size[0], 3)
        )

    def deinit_ui(self):
        super().deinit_ui()
