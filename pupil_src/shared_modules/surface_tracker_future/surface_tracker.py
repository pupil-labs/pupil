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
# TODO save surface definitions somewhere!

class Surface_Tracker_Future(Plugin):
    icon_chr = chr(0xec07)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        marker_min_perimeter=60,
        marker_min_confidence=0.6,
        invert_markers=False,
    ):
        super().__init__(g_pool)
        self.surfaces = []
        self.markers = []
        self.markers_dict = []
        self._edit_surf_verts = []
        self._last_mouse_pos = (0., 0.)
        self.gui = gui.GUI(self)

        self.marker_min_perimeter = marker_min_perimeter
        self.marker_min_confidence = marker_min_confidence
        self.inverted_markers = invert_markers

        self.menu = None
        self.button = None
        self.add_button = None

        # TODO Is anything need beyond here?

        self.robust_detection = True
        # plugin state
        self.running = True
        self.locate_3d = False # TODO Support 3D localization?

    @property
    def camera_model(self):
        return self.g_pool.capture.intrinsics

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

        self.load_surface_definitions_from_file()
        self.update_ui()

    def update_ui(self):
        self.menu.elements[:] = []
        self.menu.append(
            pyglui.ui.Info_Text(
                "This plugin detects and tracks fiducial markers visible in the scene. You can define surfaces using 1 or more marker visible within the world view by clicking *add surface*. You can edit defined surfaces by selecting *Surface edit mode*."
            )
        )
        self.menu.append(pyglui.ui.Switch("robust_detection", self, label="Robust detection"))
        self.menu.append(pyglui.ui.Switch("inverted_markers", self, label="Use "
                                                                          "inverted "
                                                                       "markers"))
        self.menu.append(
            pyglui.ui.Slider("marker_min_perimeter", self, step=1, min=30, max=100)
        )
        self.menu.append(
            pyglui.ui.Slider("marker_min_confidence", self, step=0.01, min=0, max=1)
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
            s_menu.append(pyglui.ui.Button("Open Debug Window",
                                           self.gui.surface_windows[
                                               s].open_close_window))

            def make_remove_s(i):
                return lambda: self.remove_surface(i)

            remove_s = make_remove_s(idx)
            s_menu.append(pyglui.ui.Button("remove", remove_s))
            self.menu.append(s_menu)

    def load_surface_definitions_from_file(self):
        self.surface_definitions = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.user_dir, "surface_definitions")
        )

        for init_dict in self.surface_definitions.get("surfaces", []):
            self.add_surface(None, init_dict=init_dict)

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        if self.running:
            self._detect_markers(frame)

        # Update surfaces whose verticies have been changes through the GUI
        for surface, idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(idx, self._last_mouse_pos.copy(), self.camera_model)

        # TODO what exactly is turned off when not running?
        # Update surfaces and gaze on surfaces
        events["surfaces"] = []
        gaze_events = events.get("gaze", [])
        for surface in self.surfaces:
            surface.update(self.markers, self.camera_model)

            # Clean up gaze history
            while surface.gaze_history and gaze_events and gaze_events[-1]['timestamp']\
                    - \
                    surface.gaze_history[0][
                'timestamp'] > surface.gaze_history_length:
                surface.gaze_history.popleft()

            if surface.detected:
                gaze_on_srf = self._gaze_to_surf(surface, gaze_events)

                # Update gaze history
                for gaze, event in zip(gaze_on_srf, gaze_events):
                    if event['confidence'] < 0.6: # TODO is this a good threshold?
                        continue
                    surface.gaze_history.append({'timestamp': event['timestamp'],
                                                 'gaze': gaze})

                if self.g_pool.app != "player":
                    surface._generate_heatmap()

                fixation_events = events.get("fixations", [])
                fixations_on_srf = self._gaze_to_surf(surface, fixation_events)

                surface_event = {
                    "topic": "surfaces.{}".format(surface.name),
                    "name": surface.name,
                    # "uid": s.id, # TODO correctly fix issue when saving surfaces with
                    #  the same name!
                    "m_to_screen": surface._surf_to_dist_img_trans.tolist(),
                    "m_from_screen": surface._dist_img_to_surf_trans.tolist(),
                    "gaze_on_srf": gaze_on_srf,
                    "fixations_on_srf": fixations_on_srf,
                    "timestamp": frame.timestamp,
                }
                events["surfaces"].append(surface_event)

    def add_surface(self, _, init_dict=None):
        surface = Surface(self.marker_min_perimeter,
                       self.marker_min_confidence, init_dict=init_dict)
        self.surfaces.append(surface)
        self.gui.add_surface(surface)
        self.update_ui()
        self.notify_all({"subject": "surfaces_changed"})

    def remove_surface(self, i):
        self.gui.remove_surface(self.surfaces[i])
        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_ui()
        self.notify_all({"subject": "surfaces_changed"})

    def _gaze_to_surf(self, surf, gaze_events):
        result = []
        for event in gaze_events:
            norm_pos = event["norm_pos"]
            img_point = methods.denormalize(norm_pos, self.camera_model.resolution,
                                            flip_y=True) # TODO Do not denormalize?
            img_point = np.array(img_point)
            surf_point = surf.map_to_surf(img_point, self.camera_model)
            surf_point = surf_point.tolist()
            result.append(surf_point)
        return result

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
                invert_image=self.inverted_markers
            )
        else:
            markers = marker_det.detect_markers(
                gray,
                grid_size=5,
                aperture=11,
                min_marker_perimeter=self.marker_min_perimeter,
            )

        # TODO rewrite marker detection to already output marker objects
        self.markers_dict = markers
        self.markers = [Marker(
            m["id"], m["id_confidence"], m["verts"],
            m["perimeter"]) for m in markers]

    def gl_display(self):
        self.gui.update()

    def on_pos(self, pos):
        self._last_mouse_pos = np.array(pos, dtype=np.float32)

    def on_click(self, pos, button, action):
        self.gui.on_click(pos, button, action)

    def on_notify(self, notification):
        if notification["subject"] == "surfaces_changed":
            logger.info("Surfaces changed. Saving to file.")
            self.save_surface_definitions_to_file()

    def get_init_dict(self):
        return {
            "marker_min_perimeter": self.marker_min_perimeter,
            "marker_min_confidence": self.marker_min_confidence,
            "inverted_markers": self.inverted_markers,
        }

    def save_surface_definitions_to_file(self):
        self.surface_definitions["surfaces"] = [surface.save_to_dict() for surface in self.surfaces if surface.defined]
        self.surface_definitions.save()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def cleanup(self):
        self.save_surface_definitions_to_file()

        for s in self.surfaces: # TODO Is this needed?
            s.cleanup()

Marker = collections.namedtuple("Marker", ["id", "id_confidence", "verts", "perimeter"])