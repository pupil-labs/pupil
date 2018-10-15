'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import collections
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pyglui

from plugin import Plugin
import square_marker_detect as marker_det
import methods
import file_methods
from .surface import Surface
from . import gui

# TODO fo heatmap updates and exports in background

# TODO Improve marker coloring, marker toggle is barely visible
# TODO Would it be nice to have heatmap and ids be toggles rather then different modes?

class Surface_Tracker_Future(Plugin):
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
    - Trim markers change # TODO Update heatmaps on trim marker change
    - Surface parameters have changed

    """
    icon_chr = chr(0xec07)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        marker_min_perimeter=60,
        inverted_markers=False,
    ):
        super().__init__(g_pool)
        self.current_frame_idx = None
        self.surfaces = []
        self.markers = []
        self.markers_dict = []
        self._edit_surf_verts = []
        self._last_mouse_pos = (0., 0.)
        self.gui = gui.GUI(self)

        self.marker_min_perimeter = marker_min_perimeter
        self.marker_min_confidence = 0.6
        self.inverted_markers = inverted_markers

        self.robust_detection = True
        self.running = True

        self.menu = None
        self.button = None
        self.add_button = None

        self.locate_3d = False  # TODO currently not supported. Is this ok?
        self.load_surface_definitions_from_file()

    @property
    def camera_model(self):
        return self.g_pool.capture.intrinsics

    @property
    def Surface_Class(self):
        return Surface

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Surface Tracker"

        self.button = pyglui.ui.Thumb("running", self, label="S", hotkey="s")
        self.button.on_color[:] = (.1, .2, 1., .8)
        self.g_pool.quickbar.append(self.button)
        self.add_button = pyglui.ui.Thumb(
            "add_surface",
            setter=self.add_surface,
            getter=lambda: False,
            label="A",
            hotkey="a",
        )
        self.g_pool.quickbar.append(self.add_button)
        self.update_ui()


    def update_ui(self):
        try:
            self.menu.elements[:] = []
        except AttributeError:
            return
        self.menu.append(
            pyglui.ui.Info_Text(
                "This plugin detects and tracks fiducial markers visible in the scene. You can define surfaces using 1 or more marker visible within the world view by clicking *add surface*. You can edit defined surfaces by selecting *Surface edit mode*."
            )
        )
        self.menu.append(
            pyglui.ui.Switch("robust_detection", self, label="Robust detection")
        )
        self.menu.append(
            pyglui.ui.Switch(
                "inverted_markers", self, label="Use " "inverted " "markers"
            )
        )
        self.menu.append(
            pyglui.ui.Slider("marker_min_perimeter", self, step=1, min=30, max=100)
        )
        self.menu.append(pyglui.ui.Switch("locate_3d", self, label="3D localization"))
        self.menu.append(
            pyglui.ui.Selector(
                "state",
                self.gui,
                label="Mode",
                labels=[e.value for e in gui.State],
                selection=[e for e in gui.State],
            )
        )
        self.menu.append(pyglui.ui.Button("Add surface", lambda: self.add_surface("_")))

        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = pyglui.ui.Growing_Menu("Surface {}".format(idx))
            s_menu.collapsed = True
            s_menu.append(pyglui.ui.Text_Input("name", s))
            s_menu.append(pyglui.ui.Text_Input("x", s.real_world_size, label="X size"))
            s_menu.append(pyglui.ui.Text_Input("y", s.real_world_size, label="Y size"))
            s_menu.append(
                pyglui.ui.Text_Input(
                    "gaze_history_length", s, label="Gaze History Length [seconds]"
                )
            )
            s_menu.append(
                pyglui.ui.Button(
                    "Open Debug Window", self.gui.surface_windows[s].open_close_window
                )
            )

            def make_remove_s(i):
                return lambda: self.remove_surface(i)

            remove_s = make_remove_s(idx)
            s_menu.append(pyglui.ui.Button("remove", remove_s))
            self.menu.append(s_menu)

    def load_surface_definitions_from_file(self):
        surface_definitions = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.user_dir, "surface_definitions")
        )

        for init_dict in surface_definitions.get("surfaces", []):
            self.add_surface(None, init_dict=init_dict)

    # TODO online specific
    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        if self.running:
            self._detect_markers(frame)

        self._update_surface_locations(frame.index)
        self._update_surface_corners()
        self._add_surface_events(events, frame)
        self._update_surface_gaze_history(events, frame.timestamp)

        if self.gui.state == gui.State.SHOW_HEATMAP:
            self._update_surface_heatmaps()

    def _update_surface_locations(self, idx):
        for surface in self.surfaces:
            surface.update_location(idx, self.markers, self.camera_model)

    def _add_surface_events(self, events, frame):
        events["surfaces"] = []
        for surface in self.surfaces:
            if surface.detected:
                gaze_events = events.get("gaze", [])
                gaze_on_surf = surface.map_events(gaze_events, self.camera_model)
                fixation_events = events.get("fixations", [])
                fixations_on_surf = surface.map_events(fixation_events, self.camera_model)

                surface_event = {
                    "topic": "surfaces.{}".format(surface.name),
                    "name": surface.name,
                    "uid": surface.uid,
                    "surf_to_img_trans": surface.surf_to_img_trans.tolist(),
                    "img_to_surf_trans": surface.img_to_surf_trans.tolist(),
                    "gaze_on_surf": gaze_on_surf,
                    "fixations_on_surf": fixations_on_surf,
                    "timestamp": frame.timestamp,
                }
                events["surfaces"].append(surface_event)

    def _update_surface_gaze_history(self, events, world_timestamp):
        surfaces_gaze_dict = {e['uid']: e['gaze_on_surf'] for e in events["surfaces"]}

        for surface in self.surfaces:
            try:
                surface.update_gaze_history(surfaces_gaze_dict[surface.uid], world_timestamp)
            except KeyError:
                pass

    def _update_surface_corners(self):
        for surface, corner_idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(corner_idx, self._last_mouse_pos.copy(), self.camera_model)

    # TODO Online specific
    def _update_surface_heatmaps(self):
        for surface in self.surfaces:
            surface.update_heatmap()

    def add_surface(self, _, init_dict=None):
        if self.markers or init_dict is not None:
            surface = self.Surface_Class(on_surface_changed=self.on_surface_change, init_dict=init_dict)
            self.surfaces.append(surface)
            self.gui.add_surface(surface)
            self.update_ui()
        else:
            logger.warning("Can not add a new surface: No markers found in the image!")

    def remove_surface(self, i):
        self.gui.remove_surface(self.surfaces[i])
        del self.surfaces[i]
        self.update_ui()
        self.save_surface_definitions_to_file()

    def _detect_markers(self, frame):
        gray = frame.gray
        if self.inverted_markers:
            gray = 255 - gray

        if self.robust_detection:
            markers = marker_det.detect_markers_robust(
                gray,
                grid_size=5,
                aperture=11,
                prev_markers=self.markers_dict,
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
        self.markers_dict = markers
        markers = [
            Marker(m["id"], m["id_confidence"], m["verts"], m["perimeter"])
            for m in markers
        ]
        self.markers = self._filter_markers(markers)

    def _filter_markers(self, markers):
        filtered_markers = [
            m
            for m in markers
            if m.perimeter >= self.marker_min_perimeter
            and m.id_confidence > self.marker_min_confidence
        ]

        # if an id shows twice use the bigger marker (usually this is a screen camera echo artifact.)
        marker_by_id = {}
        for m in filtered_markers:
            if not m.id in marker_by_id or m.perimeter > marker_by_id[m.id].perimeter:
                marker_by_id[m.id] = m

        return list(marker_by_id.values())

    def gl_display(self):
        self.gui.update()

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
            os.path.join(self.g_pool.user_dir, "surface_definitions")
        )
        surface_definitions["surfaces"] = [
            surface.save_to_dict() for surface in self.surfaces if surface.defined
        ]
        surface_definitions.save()

    def on_surface_change(self, surface):
        self.save_surface_definitions_to_file()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def cleanup(self):
        self.save_surface_definitions_to_file()


Marker = collections.namedtuple("Marker", ["id", "id_confidence", "verts", "perimeter"])
