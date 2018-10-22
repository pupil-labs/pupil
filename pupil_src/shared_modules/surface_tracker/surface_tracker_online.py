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


from surface_tracker import Heatmap_Mode
from surface_tracker.surface_tracker import Surface_Tracker
from surface_tracker.surface_online import Surface_Online


class Surface_Tracker_Online(Surface_Tracker):
    def __init__(self, g_pool, marker_min_perimeter=60, inverted_markers=False):
        self.Surface_Class = Surface_Online
        super().__init__(
            g_pool,
            marker_min_perimeter=marker_min_perimeter,
            inverted_markers=inverted_markers,
        )
        self.ui_info_text = "This plugin detects and tracks fiducial markers visible in the scene. You can define surfaces using 1 or more marker visible within the world view by clicking *add surface*. You can edit defined surfaces by selecting *Surface edit mode*."
        self.supported_heatmap_modes = [Heatmap_Mode.WITHIN_SURFACE]
        self.running = True

        self.menu = None
        self.button = None
        self.add_button = None

        self.locate_3d = False  # TODO currently not supported. Is this ok?

    @property
    def save_dir(self):
        return self.g_pool.user_dir

    def init_ui(self):
        super().init_ui()

        self.button = pyglui.ui.Thumb("running", self, label="S", hotkey="s")
        self.button.on_color[:] = (.1, .2, 1., .8)
        self.g_pool.quickbar.append(self.button)

    def per_surface_ui(self, surface):
        def set_name(val):
            surface.name = val
            self.notify_all(
                {
                    "subject": "surface_tracker.surface_name_changed.{}".format(
                        surface.name
                    ),
                    "uid": surface.uid,
                }
            )

        def set_x(val):
            if val <= 0:
                logger.warning("Surface size must be positive!")
            surface.real_world_size["x"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed.{}".format(
                        surface.name
                    ),
                    "uid": surface.uid,
                }
            )

        def set_y(val):
            if val <= 0:
                logger.warning("Surface size must be positive!")
            surface.real_world_size["y"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed.{}".format(
                        surface.name
                    ),
                    "uid": surface.uid,
                }
            )

        def set_gaze_hist_len(val):
            if val <= 0:
                logger.warning("Gaze history length must be a positive number!")
                return
            surface.gaze_history_length = val

        def set_hm_smooth(val):
            if val < 1:
                logger.warning("Heatmap SMoothness must be in (1,200)!")
                return
            surface._heatmap_scale_inv = val
            surface.heatmap_scale = 201 - val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed.{}".format(
                        surface.name
                    ),
                    "uid": surface.uid,
                    "delay": 0.5,
                }
            )

        idx = self.surfaces.index(surface)
        s_menu = pyglui.ui.Growing_Menu("{}".format(self.surfaces[idx].name))
        s_menu.collapsed = True
        s_menu.append(pyglui.ui.Text_Input("name", surface, setter=set_name))
        s_menu.append(
            pyglui.ui.Text_Input(
                "x", surface.real_world_size, label="X size", setter=set_x
            )
        )
        s_menu.append(
            pyglui.ui.Text_Input(
                "y", surface.real_world_size, label="Y size", setter=set_y
            )
        )
        s_menu.append(
            pyglui.ui.Text_Input(
                "gaze_history_length",
                surface,
                label="Gaze History Length [seconds]",
                setter=set_gaze_hist_len,
            )
        )
        s_menu.append(
            pyglui.ui.Slider(
                "_heatmap_scale_inv",
                surface,
                label="Heatmap Smoothness",
                setter=set_hm_smooth,
                step=1,
                min=1,
                max=200,
            )
        )
        s_menu.append(
            pyglui.ui.Button(
                "Open Surface in Window",
                self.gui.surface_windows[surface].open_close_window,
            )
        )
        remove_surf = lambda: self.remove_surface(idx)
        s_menu.append(pyglui.ui.Button("remove", remove_surf))
        self.menu.append(s_menu)

    def recent_events(self, events):
        super().recent_events(events)

        if not self.current_frame:
            return

        self._update_surface_gaze_history(events, self.current_frame.timestamp)

        if self.gui.show_heatmap:
            self._update_surface_heatmaps()

    def update_markers(self, frame):
        if self.running:
            self._detect_markers(frame)

    def _update_surface_locations(self, idx):
        for surface in self.surfaces:
            surface.update_location(idx, self.markers, self.camera_model)

    def _update_surface_gaze_history(self, events, world_timestamp):
        surfaces_gaze_dict = {e["uid"]: e["gaze_on_surf"] for e in events["surfaces"]}

        for surface in self.surfaces:
            try:
                surface.update_gaze_history(
                    surfaces_gaze_dict[surface.uid], world_timestamp
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
            surface.update_heatmap()

    def deinit_ui(self):
        super().deinit_ui()
        self.g_pool.quickbar.remove(self.button)
        self.button = None
