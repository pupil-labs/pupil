import collections

import numpy as np
import pyglui

from plugin import Plugin
import square_marker_detect as marker_det
import methods
from .surface import Surface
from . import gui

# TODO save surface definitions somewhere!

class Surface_Tracker_Future(Plugin):
    icon_chr = chr(0xec07)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        min_marker_perimeter=60,
        invert_image=False,
    ):
        super().__init__(g_pool)
        self.surfaces = []
        self.markers = []
        self.markers_dict = []
        self.camera_model = g_pool.capture.intrinsics
        self.marker_min_perimeter = min_marker_perimeter
        self.marker_min_confidence = 0.0  # TODO is this set anywhere?

        self.gui = gui.GUI(self)

        # TODO Is anything need beyond here?


        self.order = .2


        # self.load_surface_definitions_from_file()

        # plugin state
        self.running = True

        self.robust_detection = True
        self.aperture = 11

        self.invert_image = invert_image

        self.img_shape = None

        self.locate_3d = False

        self.menu = None
        self.button = None
        self.add_button = None

    def load_surface_definitions_from_file(self):
        pass
        # TODO Revamp
        # # all registered surfaces
        # self.surface_definitions = Persistent_Dict(
        #     os.path.join(self.g_pool.user_dir, "surface_definitions")
        # )
        # self.surfaces = [
        #     Reference_Surface(self.g_pool, saved_definition=d)
        #     for d in self.surface_definitions.get("realtime_square_marker_surfaces", [])
        # ]

    def on_notify(self, notification):
        # TODO revamp
        pass
        # if notification["subject"] == "surfaces_changed":
        #     logger.info("Surfaces changed. Saving to file.")
        #     self.save_surface_definitions_to_file()

    def add_surface(self, _):
        surf = Surface(self.g_pool.capture.intrinsics, self.marker_min_perimeter,
                       self.marker_min_confidence)
        self.surfaces.append(surf)
        self.update_gui_markers()

    def remove_surface(self, i):
        remove_surface = self.surfaces[i]
        if remove_surface == self.marker_edit_surface:
            self.marker_edit_surface = None
        if remove_surface in self.edit_surfaces:
            self.edit_surfaces.remove(remove_surface)

        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_gui_markers()
        self.notify_all({"subject": "surfaces_changed"})

    def init_ui(self):
        # TODO revamp
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
        self.update_gui_markers()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def update_gui_markers(self):
        # TODO revamp
        self.menu.elements[:] = []
        self.menu.append(
            pyglui.ui.Info_Text(
                "This plugin detects and tracks fiducial markers visible in the scene. You can define surfaces using 1 or more marker visible within the world view by clicking *add surface*. You can edit defined surfaces by selecting *Surface edit mode*."
            )
        )
        self.menu.append(pyglui.ui.Switch("robust_detection", self, label="Robust detection"))
        self.menu.append(pyglui.ui.Switch("invert_image", self, label="Use inverted markers"))
        self.menu.append(
            pyglui.ui.Slider("marker_min_perimeter", self, step=1, min=30, max=100)
        )
        self.menu.append(pyglui.ui.Switch("locate_3d", self, label="3D localization"))
        self.menu.append(
            pyglui.ui.Selector(
                "state",
                self.gui,
                label="Mode",
                labels=[
                    "Show Markers and Surfaces",
                    "Show marker IDs",
                    "Show Heatmaps",
                ],
                selection=[
                    gui.State.SHOW_SURF,
                    gui.State.SHOW_IDS,
                    gui.State.SHOW_HEATMAP
                ],
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
            # s_menu.append(
            #     pyglui.ui.Text_Input(
            #         "gaze_history_length", s, label="Gaze History Length [seconds]"
            #     )
            # )
            # s_menu.append(pyglui.ui.Button("Open Debug Window", s.open_close_window))
            # closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)

            remove_s = make_remove_s(idx)
            s_menu.append(pyglui.ui.Button("remove", remove_s))
            self.menu.append(s_menu)

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return
        self.img_shape = frame.height, frame.width, 3

        if self.running:
            self._detect_markers(frame)

        # TODO what exactly is turned off when not running?
        events["surfaces"] = []
        for surface in self.surfaces:
            surface.update(self.markers)

            if surface.detected:
                gaze_events = events.get("gaze", [])
                gaze_on_srf = self._gaze_to_surf(surface, gaze_events)
                fixation_events = events.get("fixations", [])
                fixations_on_srf = self._gaze_to_surf(surface, fixation_events)

                surface_event = {
                    "topic": "surfaces.{}".format(surface.name),
                    "name": surface.name,
                    # "uid": s.id, # TODO correctly fix issue when saving surfaces with
                    #  the same name!
                    "m_to_screen": surface.surf_to_img_trans.tolist(),
                    "m_from_screen": surface.img_to_surf_trans.tolist(),
                    "gaze_on_srf": gaze_on_srf,
                    "fixations_on_srf": fixations_on_srf,
                    "timestamp": frame.timestamp,
                }
                events["surfaces"].append(surface_event)

    def _gaze_to_surf(self, surf, gaze_events):
        result = []
        for event in gaze_events:
            norm_pos = event["norm_pos"]
            img_point = methods.denormalize(norm_pos, self.camera_model.resolution,
                                            flip_y=True)
            img_point = np.array(img_point)
            surf_point = surf.map_to_surf(img_point)
            surf_point = surf_point.tolist()
            result.append(surf_point)
        return result

    def _detect_markers(self, frame):
        gray = frame.gray
        if self.invert_image:
            gray = 255 - gray

        if self.robust_detection:
            markers = marker_det.detect_markers_robust(
                gray, grid_size=5, aperture=self.aperture,
                prev_markers=self.markers_dict,
                true_detect_every_frame=3,
                min_marker_perimeter=self.marker_min_perimeter)
        else:
            markers = marker_det.detect_markers(
                gray, grid_size=5, aperture=self.aperture,
                min_marker_perimeter=self.marker_min_perimeter)

        # TODO rewrite marker detection to already output marker objects
        self.markers_dict = markers
        self.markers = [Marker(
            m["id"], m["id_confidence"], m["verts"],
            m["perimeter"]) for m in markers]

    def get_init_dict(self):
        return {
            "mode": self.mode,
            "min_marker_perimeter": self.marker_min_perimeter,
            "invert_image": self.invert_image,
            "robust_detection": self.robust_detection,
        }

    def gl_display(self):
        self.gui.update()

    def on_pos(self, pos):
        self.gui.on_pos(pos)

    def on_click(self, pos, button, action):
        self.gui.on_click(pos, button, action)

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.save_surface_definitions_to_file()

        for s in self.surfaces:
            s.cleanup()

Marker = collections.namedtuple("Marker", ["id", "id_confidence", "verts", "perimeter"])