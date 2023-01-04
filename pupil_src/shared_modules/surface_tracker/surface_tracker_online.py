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

logger = logging.getLogger(__name__)

import gl_utils
import pyglui
import pyglui.cygl.utils as pyglui_utils

from .gui import Heatmap_Mode
from .surface_online import Surface_Online
from .surface_tracker import Surface_Tracker


class Surface_Tracker_Online(Surface_Tracker):
    """
    The Surface_Tracker_Online does marker based AOI tracking in real-time. All
    necessary computation is done per frame.
    """

    def __init__(self, g_pool, *args, **kwargs):
        self.freeze_scene = False
        self.frozen_scene_frame = None
        self.frozen_scene_tex = None
        super().__init__(g_pool, *args, use_online_detection=True, **kwargs)

        self.menu = None
        self.button = None
        self.add_button = None

    @property
    def Surface_Class(self):
        return Surface_Online

    @property
    def _save_dir(self):
        return self.g_pool.user_dir

    @property
    def has_freeze_feature(self):
        return True

    @property
    def ui_info_text(self):
        return (
            "This plugin detects and tracks fiducial markers visible in the "
            "scene. You can define surfaces using 1 or more marker visible within"
            " the world view by clicking *add surface*. You can edit defined "
            "surfaces by selecting *Surface edit mode*."
        )

    @property
    def supported_heatmap_modes(self):
        return [Heatmap_Mode.WITHIN_SURFACE]

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

    def _per_surface_ui_custom(self, surface, surf_menu):
        def set_gaze_hist_len(val):
            if val <= 0:
                logger.warning("Gaze history length must be a positive number!")
                return
            surface.gaze_history_length = val

        surf_menu.append(
            pyglui.ui.Text_Input(
                "gaze_history_length",
                surface,
                label="Gaze History Length [seconds]",
                setter=set_gaze_hist_len,
            )
        )

    def recent_events(self, events):
        if self._ui_heatmap_mode_selector is not None:
            self._ui_heatmap_mode_selector.read_only = True
        if self.freeze_scene:
            # If frozen, we overwrite the frame event with the last frame we have saved
            current_frame = events.get("frame")
            events["frame"] = self.current_frame

        super().recent_events(events)

        if not self.current_frame:
            return

        self._update_surface_gaze_history(events, self.current_frame.timestamp)

        if self.gui.show_heatmap:
            self._update_surface_heatmaps()

        if self.freeze_scene:
            # After we are done, we put the actual current_frame back, so other
            # plugins can access it.
            events["frame"] = current_frame

    def _update_markers(self, frame):
        self._detect_markers(frame)

    def _update_surface_locations(self, frame_index):
        for surface in self.surfaces:
            surface.update_location(frame_index, self.markers, self.camera_model)

    def _update_surface_corners(self):
        for surface, corner_idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(
                    corner_idx, self._last_mouse_pos.copy(), self.camera_model
                )

    def _update_surface_heatmaps(self):
        for surface in self.surfaces:
            gaze_on_surf = surface.gaze_history
            gaze_on_surf = (
                g
                for g in gaze_on_surf
                if g["confidence"] >= self.g_pool.min_data_confidence
            )
            gaze_on_surf = list(gaze_on_surf)
            surface.update_heatmap(gaze_on_surf)

    def _update_surface_gaze_history(self, events, world_timestamp):
        surfaces_gaze_dict = {
            e["name"]: e["gaze_on_surfaces"] for e in events["surfaces"]
        }

        for surface in self.surfaces:
            try:
                surface.update_gaze_history(
                    surfaces_gaze_dict[surface.name], world_timestamp
                )
            except KeyError:
                pass

    def on_add_surface_click(self, _=None):
        if self.freeze_scene:
            logger.warning("Surfaces cannot be added while the scene is frozen!")
        else:
            # NOTE: This is slightly different than the super() implementation.
            # We need to save the surface definition after adding it, but the Surface
            # Store does not store undefined surfaces. Therefore, we need to call
            # surface.update_location() once. This will define the surface and allow us
            # to save it.
            if self.markers and self.current_frame is not None:
                surface = self.Surface_Class(name=f"Surface {len(self.surfaces) + 1}")
                self.add_surface(surface)
                surface.update_location(
                    self.current_frame.index, self.markers, self.camera_model
                )
                self.save_surface_definitions_to_file()
            else:
                logger.warning(
                    "Can not add a new surface: No markers found in the image!"
                )

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
