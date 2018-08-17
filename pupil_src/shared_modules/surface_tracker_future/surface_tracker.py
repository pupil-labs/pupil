from enum import Enum
import collections

import numpy as np
import cv2

from plugin import Plugin
import square_marker_detect as marker_det
import methods
from .surface import Surface

# GUI
import pyglui
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl

# TODO save surface definitions somewhere!

class Surface_Tracker_Future(Plugin):
    icon_chr = chr(0xec07)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        mode="Show Markers and Surfaces",
        min_marker_perimeter=60,
        invert_image=False,
    ):
        super().__init__(g_pool)
        self.surfaces = []
        self.markers = []
        self.markers_dict = []
        self.gui_state = GUI_State.SHOW_SURF
        self.camera_model = g_pool.capture.intrinsics


        self.order = .2


        self.load_surface_definitions_from_file()

        # edit surfaces
        self.edit_surfaces = []
        self.edit_surf_verts = []
        self.marker_edit_surface = None
        # plugin state
        self.mode = mode
        self.running = True

        self.robust_detection = True
        self.aperture = 11
        self.marker_min_perimeter = min_marker_perimeter
        self.marker_min_confidence = 0.0 # TODO is this set anywhere?
        self.locate_3d = False
        self.invert_image = invert_image

        self.img_shape = None
        self._last_mouse_pos = 0, 0

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

    def on_pos(self, pos):
        self._last_mouse_pos = pos

    def on_click(self, pos, button, action):
        # TODO revamp
        pass
        # if self.mode == "Show Markers and Surfaces":
        #     if action == GLFW_PRESS:
        #         for s in self.surfaces:
        #             toggle = s.get_mode_toggle(pos, self.img_shape)
        #             if toggle == "surface_mode":
        #                 if s in self.edit_surfaces:
        #                     self.edit_surfaces.remove(s)
        #                 else:
        #                     self.edit_surfaces.append(s)
        #             elif toggle == "marker_mode":
        #                 if self.marker_edit_surface == s:
        #                     self.marker_edit_surface = None
        #                 else:
        #                     self.marker_edit_surface = s
        #
        #     if action == GLFW_RELEASE:
        #         if self.edit_surf_verts:
        #             # if we had draged a vertex lets let other know the surfaces changed.
        #             self.notify_all({"subject": "surfaces_changed", "delay": 2})
        #         self.edit_surf_verts = []
        #
        #     elif action == GLFW_PRESS:
        #         surf_verts = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
        #         x, y = pos
        #         for s in self.edit_surfaces:
        #             if s.detected and s.defined:
        #                 for (vx, vy), i in zip(
        #                     s.surface_to_img(np.array(surf_verts)), range(4)
        #                 ):
        #                     if sqrt((x - vx) ** 2 + (y - vy) ** 2) < 15:  # img pixels
        #                         self.edit_surf_verts.append((s, i))
        #                         return
        #
        #         if self.marker_edit_surface:
        #             for m in self.markers:
        #                 if m["perimeter"] >= self.min_marker_perimeter:
        #                     vx, vy = m["centroid"]
        #                     if sqrt((x - vx) ** 2 + (y - vy) ** 2) < 15:
        #                         if m["id"] in self.marker_edit_surface.markers:
        #                             self.marker_edit_surface.remove_marker(m)
        #                             self.notify_all(
        #                                 {"subject": "surfaces_changed", "delay": 1}
        #                             )
        #                         else:
        #                             self.marker_edit_surface.add_marker(
        #                                 m,
        #                                 self.markers,
        #                                 self.min_marker_perimeter,
        #                                 self.min_id_confidence,
        #                             )
        #                             self.notify_all(
        #                                 {"subject": "surfaces_changed", "delay": 1}
        #                             )

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
                "mode",
                self,
                label="Mode",
                selection=[
                    "Show Markers and Surfaces",
                    "Show marker IDs",
                    "Show Heatmaps",
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
        for s in self.surfaces:
            s.update(self.markers)

            if s.detected:
                gaze_events = events.get("gaze", [])
                gaze_on_srf = self._gaze_to_surf(s, gaze_events)
                fixation_events = events.get("fixations", [])
                fixations_on_srf = self._gaze_to_surf(s, fixation_events)

                surface_event = {
                    "topic": "surfaces.{}".format(s.name),
                    "name": s.name,
                    # "uid": s.id, # TODO correctly fix issue when saving surfaces with
                    #  the same name!
                    "m_to_screen": s.surf_to_img_trans.tolist(),
                    "m_from_screen": s.img_to_surf_trans.tolist(),
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
        self._draw_markers()

        if self.gui_state in [GUI_State.SHOW_SURF]:
            self._draw_surface_frames()

        # if self.mode == "Show Markers and Surfaces":
        #     for m in self.markers:
        #         hat = np.array(
        #             [[[0, 0], [0, 1], [.5, 1.3], [1, 1], [1, 0], [0, 0]]],
        #             dtype=np.float32,
        #         )
        #         hat = cv2.perspectiveTransform(hat, m_marker_to_screen(m))
        #         if (
        #             m["perimeter"] >= self.min_marker_perimeter
        #             and m["id_confidence"] > self.min_id_confidence
        #         ):
        #             draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 1., 1., .5))
        #             draw_polyline(
        #                 hat.reshape((6, 2)),
        #                 color=RGBA(0.1, 1., 1., .3),
        #                 line_type=GL_POLYGON,
        #             )
        #         else:
        #             draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 1., 1., .5))
        #
        #     for s in self.surfaces:
        #         if s not in self.edit_surfaces and s is not self.marker_edit_surface:
        #             s.gl_draw_frame(self.img_shape)
        #
        #     for s in self.edit_surfaces:
        #         s.gl_draw_frame(self.img_shape, highlight=True, surface_mode=True)
        #         s.gl_draw_corners()
        #
        #     if self.marker_edit_surface:
        #         inc = []
        #         exc = []
        #         for m in self.markers:
        #             if m["perimeter"] >= self.min_marker_perimeter:
        #                 if m["id"] in self.marker_edit_surface.markers:
        #                     inc.append(m["centroid"])
        #                 else:
        #                     exc.append(m["centroid"])
        #         draw_points(exc, size=20, color=RGBA(1., 0.5, 0.5, .8))
        #         draw_points(inc, size=20, color=RGBA(0.5, 1., 0.5, .8))
        #         self.marker_edit_surface.gl_draw_frame(
        #             self.img_shape,
        #             color=(0.0, 0.9, 0.6, 1.0),
        #             highlight=True,
        #             marker_mode=True,
        #         )
        #
        # elif self.mode == "Show Heatmaps":
        #     for s in self.surfaces:
        #         if self.g_pool.app != "player":
        #             s.generate_heatmap()
        #         s.gl_display_heatmap()
        #
        # for s in self.surfaces:
        #     if self.locate_3d:
        #         s.gl_display_in_window_3d(self.g_pool.image_tex)
        #     else:
        #         s.gl_display_in_window(self.g_pool.image_tex)

    def _draw_markers(self):
        color = pyglui_utils.RGBA(0.1, 1., 1., .5)
        for m in self.markers:
            # TODO Update to new marker class
            # TODO Marker.verts has shape (N,1,2), change to (N,2)
            hat = np.array([[[0, 0], [0, 1], [.5, 1.3], [1, 1], [1, 0], [0, 0]]],
                           dtype=np.float32)
            hat = cv2.perspectiveTransform(hat, self.get_marker_to_img_trans(m))

            if m.perimeter >= self.marker_min_perimeter and m.id_confidence > self.marker_min_confidence:
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color)
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color,
                              line_type=gl.GL_POLYGON)
            else:
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color)

    def get_marker_to_img_trans(self, marker):
        norm_corners = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        return cv2.getPerspectiveTransform(norm_corners,
                                           np.array(marker.verts, dtype=np.float32))

    def _draw_surface_frames(self):
        r, g, b, a = (1.0, 0.2, 0.6, 1.0)

        for s in self.surfaces:
            if not s.detected:
                continue

            frame = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)
            frame = s.map_from_surf(frame)

            hat = np.array([[.3, .7], [.7, .7], [.5, .9], [.3, .7]], dtype=np.float32)
            hat = s.map_from_surf(hat)

            alpha = min(1, s.build_up_status)

            pyglui_utils.draw_polyline(frame.reshape((5, 2)), 1,
                                       pyglui_utils.RGBA(r, g, b, a * alpha))
            pyglui_utils.draw_polyline(hat.reshape((4, 2)), 1,
                                       pyglui_utils.RGBA(r, g, b, a * alpha))
            # text_anchor = frame.reshape((5, -1))[2]
            # text_anchor = text_anchor[0], text_anchor[1] - 75
            # surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
            # marker_edit_anchor = text_anchor[0], text_anchor[1] + 50
            # if self.defined:
            #     if marker_mode:
            #         pyglui_utils.draw_points([marker_edit_anchor],
            #                                  color=pyglui_utils.RGBA(0, .8, .7))
            #     else:
            #         pyglui_utils.draw_points([marker_edit_anchor])
            #     if surface_mode:
            #         pyglui_utils.draw_points([surface_edit_anchor],
            #                                  color=pyglui_utils.RGBA(0, .8, .7))
            #     else:
            #         pyglui_utils.draw_points([surface_edit_anchor])
            #
            #     self.glfont.set_blur(3.9)
            #     self.glfont.set_color_float((0, 0, 0, .8))
            #     self.glfont.draw_text(
            #         text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
            #     )
            #     self.glfont.draw_text(
            #         surface_edit_anchor[0] + 15,
            #         surface_edit_anchor[1] + 6,
            #         "edit surface",
            #     )
            #     self.glfont.draw_text(
            #         marker_edit_anchor[0] + 15,
            #         marker_edit_anchor[1] + 6,
            #         "add/remove markers",
            #     )
            #     self.glfont.set_blur(0.0)
            #     self.glfont.set_color_float((0.1, 8., 8., .9))
            #     self.glfont.draw_text(
            #         text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
            #     )
            #     self.glfont.draw_text(
            #         surface_edit_anchor[0] + 15,
            #         surface_edit_anchor[1] + 6,
            #         "edit surface",
            #     )
            #     self.glfont.draw_text(
            #         marker_edit_anchor[0] + 15,
            #         marker_edit_anchor[1] + 6,
            #         "add/remove markers",
            #     )
            # else:
            #     progress = (self.build_up_status / float(
            #         self.required_build_up)) * 100
            #     progress_text = "%.0f%%" % progress
            #     self.glfont.set_blur(3.9)
            #     self.glfont.set_color_float((0, 0, 0, .8))
            #     self.glfont.draw_text(
            #         text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
            #     )
            #     self.glfont.draw_text(
            #         surface_edit_anchor[0] + 15,
            #         surface_edit_anchor[1] + 6,
            #         "Learning affiliated markers...",
            #     )
            #     self.glfont.draw_text(
            #         marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6,
            #         progress_text
            #     )
            #     self.glfont.set_blur(0.0)
            #     self.glfont.set_color_float((0.1, 8., 8., .9))
            #     self.glfont.draw_text(
            #         text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
            #     )
            #     self.glfont.draw_text(
            #         surface_edit_anchor[0] + 15,
            #         surface_edit_anchor[1] + 6,
            #         "Learning affiliated markers...",
            #     )
            #     self.glfont.draw_text(
            #         marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6,
            #         progress_text
            #     )


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.save_surface_definitions_to_file()

        for s in self.surfaces:
            s.cleanup()


class GUI_State(Enum):
    SHOW_SURF = 1
    SHOW_IDS = 2
    SHOW_HEATMAP = 3

Marker = collections.namedtuple("Marker", ["id", "id_confidence", "verts", "perimeter"])