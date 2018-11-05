"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
from abc import ABCMeta, abstractmethod
import logging

logger = logging.getLogger(__name__)

import numpy as np
import pyglui

from plugin import Plugin
import square_marker_detect as marker_det
import file_methods

from surface_tracker import gui, Square_Marker_Detection


class Surface_Tracker(Plugin, metaclass=ABCMeta):
    """

    What happens on camera_model update?

    Marker cache is saved
    - While the marker bg process is running every 5 seconds
    - When the marker bg process is finishing

    Surfaces are saved when
    - They are added
    - The definition is completed
    - A surface bg process finished
    - The marker bg process finished

    Heatmaps are updated when
    - A surface bg process finished
    - Trim markers change
    - Surface parameters have changed

    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, marker_min_perimeter=60, inverted_markers=False):
        super().__init__(g_pool)

        self.current_frame = None
        self.surfaces = []
        self.markers = []
        self.markers_unfiltered = []
        self.previous_markers = []
        self._edit_surf_verts = []
        self._last_mouse_pos = (0.0, 0.0)
        self.gui = gui.GUI(self)

        self.marker_min_perimeter = marker_min_perimeter
        self.marker_min_confidence = 0.1
        self.inverted_markers = inverted_markers

        self.robust_detection = True
        self._add_surfaces_from_file()

    def _add_surfaces_from_file(self):
        surface_definitions = file_methods.Persistent_Dict(
            os.path.join(self._save_dir, "surface_definitions")
        )

        for init_dict in surface_definitions.get("surfaces", []):
            self.add_surface(init_dict=init_dict)

    @property
    def camera_model(self):
        return self.g_pool.capture.intrinsics

    @property
    @abstractmethod
    def _save_dir(self):
        """
        The directory that contains all files related to the Surface Tracker.
        Returns:

        """
        pass

    @property
    @abstractmethod
    def has_freeze_feature(self):
        pass

    @property
    @abstractmethod
    def ui_info_text(self):
        pass

    @property
    @abstractmethod
    def supported_heatmap_modes(self):
        pass

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.pretty_class_name
        self.add_button = pyglui.ui.Thumb(
            "add_surface",
            setter=self.add_surface,
            getter=lambda: False,
            label="A",
            hotkey="a",
        )
        self.g_pool.quickbar.append(self.add_button)
        self._update_ui()

    def _update_ui(self):
        def set_marker_min_perimeter(val):
            self.marker_min_perimeter = val
            self.notify_all(
                {
                    "subject": "surface_tracker.marker_min_perimeter_changed",
                    "delay": 0.5,
                }
            )

        def set_invert_image(val):
            self.inverted_markers = val
            self.notify_all(
                {"subject": "surface_tracker.marker_detection_params_changed"}
            )

        def set_robust_detection(val):
            self.robust_detection = val
            self.notify_all(
                {"subject": "surface_tracker.marker_detection_params_changed"}
            )

        try:
            # _update_ui is called when surfaces from a previous session are
            # restored. This happens before the UI is initialized, so we need to skip
            #  execution inthis case.
            self.menu.elements[:] = []
        except AttributeError:
            return
        self.menu.append(pyglui.ui.Info_Text(self.ui_info_text))
        self.menu.append(
            pyglui.ui.Switch("show_marker_ids", self.gui, label="Show Marker IDs")
        )
        self.menu.append(
            pyglui.ui.Switch("show_heatmap", self.gui, label="Show Heatmap")
        )
        self.menu.append(
            pyglui.ui.Selector(
                "heatmap_mode",
                self.gui,
                label="Heatmap Mode",
                labels=[e.value for e in self.supported_heatmap_modes],
                selection=[e for e in self.supported_heatmap_modes],
            )
        )
        self._update_ui_custom()
        advanced_menu = pyglui.ui.Growing_Menu("Marker Detection Parameters")
        advanced_menu.collapsed = True
        advanced_menu.append(
            pyglui.ui.Switch(
                "robust_detection",
                self,
                setter=set_robust_detection,
                label="Robust detection",
            )
        )
        advanced_menu.append(
            pyglui.ui.Switch(
                "inverted_markers",
                self,
                setter=set_invert_image,
                label="Use inverted markers",
            )
        )
        advanced_menu.append(
            pyglui.ui.Slider(
                "marker_min_perimeter",
                self,
                label="Min Marker Perimeter",
                setter=set_marker_min_perimeter,
                step=1,
                min=30,
                max=100,
            )
        )
        self.menu.append(advanced_menu)
        self.menu.append(pyglui.ui.Button("Add surface", self.add_surface))
        for surface in self.surfaces:
            self._per_surface_ui(surface)

    def _update_ui_custom(self):
        pass

    def _per_surface_ui(self, surface):
        def set_name(val):
            if val != surface.name:
                return

            names = [x.name for x in self.surfaces]
            if val in names:
                logger.warning("The name '{}' is already in use!".format(val))
                return

            self.notify_all(
                {
                    "subject": "surface_tracker.surface_name_changed",
                    "old_name": surface.name,
                    "new_name": val,
                }
            )
            surface.name = val

        def set_x(val):
            if val <= 0:
                logger.warning("Surface size must be positive!")
                return

            surface.real_world_size["x"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed",
                    "name": surface.name,
                }
            )

        def set_y(val):
            if val <= 0:
                logger.warning("Surface size must be positive!")
                return
            surface.real_world_size["y"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed",
                    "name": surface.name,
                }
            )

        def set_hm_smooth(val):
            surface._heatmap_scale = val
            val = 1 - val
            val *= 3
            surface._heatmap_blur_factor = max((1 - val), 0)
            res_exponent = max(val, 0.35)
            surface._heatmap_resolution = int(10 ** res_exponent)
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed",
                    "name": surface.name,
                    "delay": 0.5,
                }
            )

        idx = self.surfaces.index(surface)
        s_menu = pyglui.ui.Growing_Menu("{}".format(self.surfaces[idx].name))
        s_menu.collapsed = True
        s_menu.append(pyglui.ui.Text_Input("name", surface, setter=set_name))
        s_menu.append(
            pyglui.ui.Text_Input(
                "x", surface.real_world_size, label="Width", setter=set_x
            )
        )
        s_menu.append(
            pyglui.ui.Text_Input(
                "y", surface.real_world_size, label="Height", setter=set_y
            )
        )

        self._per_surface_ui_custom(surface, s_menu)

        s_menu.append(
            pyglui.ui.Slider(
                "_heatmap_scale",
                surface,
                label="Heatmap Smoothness",
                setter=set_hm_smooth,
                step=0.01,
                min=0,
                max=1,
            )
        )
        s_menu.append(
            pyglui.ui.Button(
                "Open Surface in Window",
                self.gui.surface_windows[surface].open_close_window,
            )
        )

        def remove_surf():
            self.remove_surface(idx)

        s_menu.append(pyglui.ui.Button("remove", remove_surf))
        self.menu.append(s_menu)

    def _per_surface_ui_custom(self, surface, s_menu):
        pass

    def recent_events(self, events):
        frame = events.get("frame")
        self.current_frame = frame
        if not frame:
            return

        self._update_markers(frame)
        self._update_surface_locations(frame.index)
        self._update_surface_corners()
        events["surfaces"] = self._create_surface_events(events, frame.timestamp)

    @abstractmethod
    def _update_markers(self, frame):
        pass

    def _detect_markers(self, frame):
        gray = frame.gray

        if self.robust_detection:
            markers = marker_det.detect_markers_robust(
                gray,
                grid_size=5,
                aperture=11,
                prev_markers=self.previous_markers,
                true_detect_every_frame=3,
                min_marker_perimeter=self.marker_min_perimeter,
                invert_image=self.inverted_markers,
            )
        else:
            markers = marker_det.detect_markers(
                gray,
                grid_size=5,
                aperture=11,
                min_marker_perimeter=self.marker_min_perimeter,
            )

        # Robust marker detection requires previous markers to be in a different format than the surface tracker.
        self.previous_markers = markers
        markers = [
            Square_Marker_Detection(
                m["id"], m["id_confidence"], m["verts"], m["perimeter"]
            )
            for m in markers
        ]
        markers = self._remove_duplicate_markers(markers)
        self.markers_unfiltered = markers
        self.markers = self._filter_markers(markers)

    def _remove_duplicate_markers(self, markers):
        # if an id shows twice use the bigger marker (usually this is a screen camera
        # echo artifact.)
        marker_by_id = {}
        for m in markers:
            if m.id not in marker_by_id or m.perimeter > marker_by_id[m.id].perimeter:
                marker_by_id[m.id] = m

        return list(marker_by_id.values())

    def _filter_markers(self, markers):
        markers = [
            m
            for m in markers
            if m.perimeter >= self.marker_min_perimeter
            and m.id_confidence > self.marker_min_confidence
        ]

        return markers

    @abstractmethod
    def _update_surface_locations(self, idx):
        pass

    @abstractmethod
    def _update_surface_corners(self):
        pass

    def _create_surface_events(self, events, timestamp):
        """
        Adds surface events to the current list of events.

        Args:
            events: Current list of events.
            timestamp: The timestamp of the current world camera frame
        """
        gaze_events = events.get("gaze", [])
        fixation_events = events.get("fixations", [])

        surface_events = []
        for surface in self.surfaces:
            if surface.detected:
                gaze_on_surf = surface.map_gaze_and_fixation_events(
                    gaze_events, self.camera_model
                )
                fixations_on_surf = surface.map_gaze_and_fixation_events(
                    fixation_events, self.camera_model
                )

                surface_event = {
                    "topic": "surfaces.{}".format(surface.name),
                    "name": surface.name,
                    "surf_to_img_trans": surface.surf_to_img_trans.tolist(),
                    "img_to_surf_trans": surface.img_to_surf_trans.tolist(),
                    "gaze_on_surf": gaze_on_surf,
                    "fixations_on_surf": fixations_on_surf,
                    "timestamp": timestamp,
                }
                surface_events.append(surface_event)

        return surface_events

    @abstractmethod
    def _update_surface_heatmaps(self):
        pass

    def gl_display(self):
        self.gui.update()

    def add_surface(self, _=None, init_dict=None):
        if self.markers or init_dict is not None:
            surface = self.Surface_Class(
                name="Surface {:}".format(len(self.surfaces) + 1), init_dict=init_dict
            )
            self.surfaces.append(surface)
            self.gui.add_surface(surface)
            self._update_ui()
        else:
            logger.warning("Can not add a new surface: No markers found in the image!")

    def remove_surface(self, i):
        self.gui.remove_surface(self.surfaces[i])
        del self.surfaces[i]
        self._update_ui()
        self.save_surface_definitions_to_file()

    def on_notify(self, notification):
        if notification["subject"] == "surface_tracker.surfaces_changed":
            logger.info("Surfaces changed. Saving to file.")
            self.save_surface_definitions_to_file()
        elif notification["subject"].startswith(
            "surface_tracker.heatmap_params_changed"
        ):
            self.save_surface_definitions_to_file()
        elif notification["subject"].startswith("surface_tracker.surface_name_changed"):
            self.save_surface_definitions_to_file()
            self._update_ui()

    def on_pos(self, pos):
        self._last_mouse_pos = np.array(pos, dtype=np.float32)

    def on_click(self, pos, button, action):
        self.gui.on_click(pos, button, action)

    def get_init_dict(self):
        return {
            "marker_min_perimeter": self.marker_min_perimeter,
            "inverted_markers": self.inverted_markers,
        }

    def save_surface_definitions_to_file(self):
        surface_definitions = file_methods.Persistent_Dict(
            os.path.join(self._save_dir, "surface_definitions")
        )
        surface_definitions["surfaces"] = [
            surface.save_to_dict() for surface in self.surfaces if surface.defined
        ]
        surface_definitions.save()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def cleanup(self):
        self.save_surface_definitions_to_file()
