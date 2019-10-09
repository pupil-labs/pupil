"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
from abc import ABCMeta, abstractmethod
import typing


import numpy as np
import pyglui

from plugin import Plugin

from . import gui
from .surface_marker_detector import (
    Surface_Marker_Detector,
    Surface_Marker_Detector_Mode,
)

from .surface_file_store import Surface_File_Store

logger = logging.getLogger(__name__)


class Surface_Tracker(Plugin, metaclass=ABCMeta):
    """
    The Surface_Tracker provides the base functionality for a plugin that does marker
    based AOI tracking.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Surface Tracker"

    def __init__(
        self,
        g_pool,
        marker_min_perimeter: int = 60,
        inverted_markers: bool = False,
        marker_detector_mode: typing.Union[
            Surface_Marker_Detector_Mode, str
        ] = Surface_Marker_Detector_Mode.APRILTAG_MARKER,
        use_online_detection: bool = False,
    ):
        super().__init__(g_pool)

        self.current_frame = None
        self.surfaces = []
        self.markers = []
        self.markers_unfiltered = []

        self._edit_surf_verts = []
        self._last_mouse_pos = (0.0, 0.0)
        self.gui = gui.GUI(self)

        if isinstance(marker_detector_mode, str):
            # Here we ensure that we pass a proper Surface_Marker_Detector_Mode
            # instance to Surface_Marker_Detector:
            marker_detector_mode = Surface_Marker_Detector_Mode(marker_detector_mode)

        # Even though the Surface_Marker_Detector class supports multiple detector
        # modes at once, we only want one mode to be active during surface tracking.
        marker_detector_modes = {marker_detector_mode}
        self._marker_min_perimeter = marker_min_perimeter
        self.marker_min_confidence = 0.0
        self.marker_detector = Surface_Marker_Detector(
            marker_detector_modes=marker_detector_modes,
            marker_min_perimeter=marker_min_perimeter,
            square_marker_inverted_markers=inverted_markers,
            square_marker_use_online_mode=use_online_detection,
        )
        self._surface_file_store = Surface_File_Store(parent_dir=self._save_dir)

        # Add surfaces from file
        for surface in self._surface_file_store.read_surfaces_from_file(
            surface_class=self.Surface_Class
        ):
            self.add_surface(surface)

    @property
    def marker_detector_modes(self) -> typing.Set[Surface_Marker_Detector_Mode]:
        return self.marker_detector.marker_detector_modes

    @marker_detector_modes.setter
    def marker_detector_modes(self, value: typing.Set[Surface_Marker_Detector_Mode]):
        self.marker_detector.marker_detector_modes = value

    @property
    def active_marker_detector_mode(self) -> Surface_Marker_Detector_Mode:
        assert len(self.marker_detector_modes) == 1
        return next(iter(self.marker_detector_modes))

    @active_marker_detector_mode.setter
    def active_marker_detector_mode(self, mode: Surface_Marker_Detector_Mode):
        self.marker_detector_modes = {mode}
        self.notify_all({"subject": "surface_tracker.marker_detection_params_changed"})

    @property
    def marker_min_perimeter(self) -> int:
        return self._marker_min_perimeter

    @marker_min_perimeter.setter
    def marker_min_perimeter(self, value: int):
        self._marker_min_perimeter = value
        self.marker_detector.marker_min_perimeter = value

    @property
    def inverted_markers(self) -> bool:
        return self.marker_detector.inverted_markers

    @inverted_markers.setter
    def inverted_markers(self, value: bool):
        self.marker_detector.inverted_markers = value

    @property
    def camera_model(self):
        return self.g_pool.capture.intrinsics

    @property
    @abstractmethod
    def Surface_Class(self):
        pass

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
            setter=self.on_add_surface_click,
            getter=lambda: False,
            label="A",
            hotkey="a",
        )
        self.g_pool.quickbar.append(self.add_button)
        self._update_ui()

    def _update_ui(self):
        try:
            # _update_ui is called when surfaces from a previous session are
            # restored. This happens before the UI is initialized, so we need to skip
            #  execution inthis case.
            self.menu.elements[:] = []
        except AttributeError:
            return

        self._update_ui_visualization_menu()
        self._update_ui_custom()
        self._update_ui_marker_detection_menu()

        self.menu.append(pyglui.ui.Button("Add surface", self.on_add_surface_click))
        for surface in self.surfaces:
            self._per_surface_ui(surface)

    def _update_ui_visualization_menu(self):
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

    def _update_ui_custom(self):
        pass

    def _update_ui_marker_detection_menu(self):
        def set_marker_min_perimeter(val):
            self.marker_min_perimeter = val
            self.notify_all(
                {
                    "subject": "surface_tracker.marker_min_perimeter_changed",
                    "delay": 0.5,
                }
            )

        def set_inverted_markers(val):
            self.inverted_markers = val
            self.notify_all(
                {"subject": "surface_tracker.marker_detection_params_changed"}
            )

        supported_surface_marker_detector_modes = (
            Surface_Marker_Detector_Mode.all_supported_cases()
        )
        supported_surface_marker_detector_modes = sorted(
            supported_surface_marker_detector_modes, key=lambda m: m.value
        )

        advanced_menu = pyglui.ui.Growing_Menu("Marker Detection Parameters")
        advanced_menu.collapsed = True
        advanced_menu.append(
            pyglui.ui.Switch(
                "inverted_markers",
                self,
                setter=set_inverted_markers,
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
        advanced_menu.append(
            pyglui.ui.Selector(
                "active_marker_detector_mode",
                self,
                label="Marker Detector Mode",
                labels=[mode.label for mode in supported_surface_marker_detector_modes],
                selection=[mode for mode in supported_surface_marker_detector_modes],
            )
        )
        self.menu.append(advanced_menu)

    def _per_surface_ui(self, surface):
        def set_name(val):
            if val == surface.name:
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
            if val == surface.real_world_size["x"]:
                return

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
            if val == surface.real_world_size["y"]:
                return

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
            if val == surface._heatmap_scale:
                return

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

        displayed_name = surface.name

        if surface.deprecated_definition:
            displayed_name = "Deprecated!! - " + displayed_name

        s_menu = pyglui.ui.Growing_Menu("{}".format(displayed_name))
        s_menu.collapsed = True

        if surface.deprecated_definition:
            s_menu.append(
                pyglui.ui.Info_Text(
                    "!!! This surface definition is old and deprecated! "
                    "Please re-define this surface for increased mapping accuracy! !!!"
                )
            )

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
            self.remove_surface(surface)

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
        markers = self.marker_detector.detect_markers(
            gray_img=frame.gray, frame_index=frame.index
        )
        markers = self._remove_duplicate_markers(markers)
        self.markers_unfiltered = markers
        self.markers = self._filter_markers(markers)

    def _remove_duplicate_markers(self, markers):
        # if an id shows twice use the bigger marker (usually this is a screen camera
        # echo artifact.)
        marker_by_uid = {}
        for m in markers:
            if (
                m.uid not in marker_by_uid
                or m.perimeter > marker_by_uid[m.uid].perimeter
            ):
                marker_by_uid[m.uid] = m

        return list(marker_by_uid.values())

    def _filter_markers(self, markers):
        markers = [
            m
            for m in markers
            if m.perimeter >= self.marker_min_perimeter
            and m.id_confidence > self.marker_min_confidence
        ]
        return markers

    @abstractmethod
    def _update_surface_locations(self, frame_index):
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
                    "gaze_on_surfaces": gaze_on_surf,
                    "fixations_on_surfaces": fixations_on_surf,
                    "timestamp": timestamp,
                }
                surface_events.append(surface_event)

        return surface_events

    @abstractmethod
    def _update_surface_heatmaps(self):
        pass

    def gl_display(self):
        self.gui.update()

    def on_add_surface_click(self, _=None):
        if self.markers:
            surface = self.Surface_Class(
                name="Surface {:}".format(len(self.surfaces) + 1)
            )
            self.add_surface(surface)
        else:
            logger.warning("Can not add a new surface: No markers found in the image!")

    def add_surface(self, surface):
        self.surfaces.append(surface)
        self.gui.add_surface(surface)
        self._update_ui()

    def remove_surface(self, surface):
        self.gui.remove_surface(surface)
        self.surfaces.remove(surface)
        self._update_ui()
        self.save_surface_definitions_to_file()

    def on_notify(self, notification):
        if notification["subject"] == "surface_tracker.surfaces_changed":
            logger.info("Surfaces changed. Saving to file.")
            self.save_surface_definitions_to_file()
        elif notification["subject"] == "surface_tracker.heatmap_params_changed":
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
            "marker_detector_mode": self.active_marker_detector_mode.value,
        }

    def save_surface_definitions_to_file(self):
        surfaces = [surface for surface in self.surfaces if surface.defined]
        self._surface_file_store.write_surfaces_to_file(surfaces=surfaces)

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def cleanup(self):
        self.save_surface_definitions_to_file()
