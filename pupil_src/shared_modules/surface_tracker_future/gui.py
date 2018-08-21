from enum import Enum

import numpy as np
import cv2
import pyglui
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl
import glfw



class GUI:
    """Custom GUI functionality for visualizing and editing surfaces"""

    def __init__(self, tracker):
        self.tracker = tracker

        self.state = State.SHOW_SURF
        self._edit_surf_corners = set()
        self._edit_surf_markers = set()
        self._edit_surf_verts = []

        self.color_primary = (1.0, 0.2, 0.6)
        self.color_secondary = (0.1, 1., 1.)
        self.color_highlight = (0, 0.8, 0.7)

        self.glfont = pyglui.pyfontstash.fontstash.Context()
        self.glfont.add_font("opensans", pyglui.ui.get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

    def update(self):
        self._draw_markers(self.tracker.markers)

        for surface in self.tracker.surfaces:
            if not surface.detected:
                continue

            if self.state == State.SHOW_SURF:
                self._draw_surface_frames(surface)

                if surface in self._edit_surf_markers:
                    self._draw_marker_toggles(surface)

                if surface in self._edit_surf_corners:
                    self._draw_surface_corner_handles(surface)

            # if self.mode == "Show Markers and Surfaces":
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

    def _draw_markers(self, markers):
        color = pyglui_utils.RGBA(*self.color_secondary, .5)
        for m in markers:
            # TODO Update to new marker class
            # TODO Marker.verts has shape (N,1,2), change to (N,2)
            hat = np.array([[[0, 0], [0, 1], [.5, 1.3], [1, 1], [1, 0], [0, 0]]],
                           dtype=np.float32)
            hat = cv2.perspectiveTransform(hat, self._get_marker_to_img_trans(m))

            if m.perimeter >= self.tracker.marker_min_perimeter and m.id_confidence > \
                    self.tracker.marker_min_confidence:
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color)
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color,
                              line_type=gl.GL_POLYGON)
            else:
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color)

    def _get_marker_to_img_trans(self, marker):
        norm_corners = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        return cv2.getPerspectiveTransform(norm_corners,
                                           np.array(marker.verts, dtype=np.float32))

    def _draw_surface_frames(self, surface):
        if not surface.detected:
            return

        frame, hat, text_anchor, surface_edit_anchor, marker_edit_anchor = self._get_surface_anchor_points(surface)
        alpha = min(1, surface.build_up_status)

        pyglui_utils.draw_polyline(frame.reshape((5, 2)), 1,
                                   color=pyglui_utils.RGBA(*self.color_primary,
                                                           alpha))
        pyglui_utils.draw_polyline(hat.reshape((4, 2)), 1,
                                   color=pyglui_utils.RGBA(*self.color_primary,
                                                           alpha))

        self._draw_surf_menu(surface, text_anchor, surface_edit_anchor, marker_edit_anchor)

    def _get_surface_anchor_points(self, surface):
        frame = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)
        frame = surface.map_from_surf(frame)

        hat = np.array([[.3, .7], [.7, .7], [.5, .9], [.3, .7]], dtype=np.float32)
        hat = surface.map_from_surf(hat)

        text_anchor = frame.reshape((5, -1))[2]
        text_anchor = text_anchor[0], text_anchor[1] - 75

        surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
        marker_edit_anchor = text_anchor[0], text_anchor[1] + 50

        return frame, hat, text_anchor, surface_edit_anchor, marker_edit_anchor

    def _draw_surf_menu(self, surface, text_anchor, surface_edit_anchor, marker_edit_anchor):
        marker_detection_status = "{}   {}/{}".format(surface.name,
                                                      surface.num_detected_markers,
                                                      len(surface.reg_markers))
        self._draw_text(
            (text_anchor[0] + 15, text_anchor[1] + 6),
            marker_detection_status,
            self.color_secondary
        )


        # If the surface is defined, draw menu buttons. Otherwise draw definition
        # progress.
        if surface.defined:

            # Buttons
            if surface in self._edit_surf_markers:
                pyglui_utils.draw_points([marker_edit_anchor],
                                         color=pyglui_utils.RGBA(*self.color_highlight))
            else:
                pyglui_utils.draw_points([marker_edit_anchor],
                                         color=pyglui_utils.RGBA(
                                             *self.color_primary))

            if surface in self._edit_surf_corners:
                pyglui_utils.draw_points([surface_edit_anchor],
                                         color=pyglui_utils.RGBA(*self.color_highlight))
            else:
                pyglui_utils.draw_points([surface_edit_anchor],
                                         color=pyglui_utils.RGBA(*self.color_primary))

            # Text
            self._draw_text(
                (surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6),
                "edit surface",
                self.color_secondary
            )
            self._draw_text(
                (marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6),
                "add/remove markers",
                self.color_secondary
            )
        else:
            progress_text = "{:.0f} %".format(surface.build_up_status * 100)

            self._draw_text(
                (surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6),
                "Learning affiliated markers...",
                self.color_secondary
            )
            self._draw_text(
                (marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6),
                progress_text,
                self.color_secondary
            )

    def _draw_text(self, loc, text, color):
        self.glfont.set_blur(3.9)
        self.glfont.set_color_float((0, 0, 0, .8,))
        self.glfont.draw_text(loc[0], loc[1], text)

        self.glfont.set_blur(0.0)
        self.glfont.set_color_float(color + (.9,))
        self.glfont.draw_text(loc[0], loc[1], text)

    def _draw_marker_toggles(self, surface):
        active_markers = []
        inactive_markers = []
        for marker in self.tracker.markers:
            if marker.perimeter < self.tracker.marker_min_perimeter:
                continue

            if not marker.id in surface.reg_markers.keys():
                continue

            centroid = np.mean(marker.verts, axis=0)
            centroid = (centroid[0, 0], centroid[0, 1])
            if surface.reg_markers[marker.id].active:
                active_markers.append(centroid)
            else:
                inactive_markers.append(centroid)

        pyglui_utils.draw_points(inactive_markers, size=20,
                                 color=pyglui_utils.RGBA(
                                     *self.color_primary, .8))
        pyglui_utils.draw_points(active_markers, size=20,
                                 color=pyglui_utils.RGBA(
                                     *self.color_highlight, .8))

    def _draw_surface_corner_handles(self, surface):
        norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
        img_corners = surface.map_from_surf(norm_corners)

        pyglui_utils.draw_points(img_corners, 20, pyglui_utils.RGBA(
            *self.color_primary, .5))

    def on_pos(self, pos):
        pos = np.array(pos, dtype=np.float32)
        for surface, idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(idx, pos)

    def on_click(self, pos, button, action):
        pos = np.array(pos, dtype=np.float32)

        if self.state == State.SHOW_SURF:

                # Menu Buttons
                if action == glfw.GLFW_PRESS:
                    for surface in self.tracker.surfaces:

                        if not surface.detected:
                            continue

                        surface_edit_pressed, marker_edit_pressed = \
                            self._check_surface_button_pressed(surface, pos)

                        if surface_edit_pressed:
                            if surface in self._edit_surf_corners:
                                self._edit_surf_corners.remove(surface)
                            else:
                                self._edit_surf_corners.add(surface)

                        if marker_edit_pressed:
                            if surface in self._edit_surf_markers:
                                self._edit_surf_markers.remove(surface)
                            else:
                                self._edit_surf_markers.add(surface)

                # Surface Corner Handles
                if action == glfw.GLFW_PRESS:
                    norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)],
                                            dtype=np.float32)

                    for surface in self._edit_surf_corners:
                        if surface.detected and surface.defined:
                            img_corners = surface.map_from_surf(norm_corners)
                            for corner, i in zip(img_corners, range(4)):
                                dist = np.linalg.norm(corner - pos)
                                if dist < 15:
                                    self._edit_surf_verts.append((surface, i))
                                    break
                elif action == glfw.GLFW_RELEASE:
                    self._edit_surf_verts = []
                # TODO Surface changed, save new definition


                # Marker Toggles
                # if action == glfw.GLFW_PRESS:
                #     for surface in self._edit_surf_markers:
                #         for marker in self.tracker.markers:
                #             if not marker.id in surface.reg_markers.keys():
                #                 continue
                #
                #             centroid = np.mean(marker.verts, axis=0)
                #             dist = np.linalg.norm(centroid - pos)
                #             if dist < 15:
                #                 surface.reg_markers[marker.id].active = False
                # TODO Surface changed, save new definition

    def _check_surface_button_pressed(self, surface, pos):
        frame, hat, text_anchor, surface_edit_anchor, marker_edit_anchor = self._get_surface_anchor_points(surface)

        surface_edit_anchor = np.array(surface_edit_anchor)
        marker_edit_anchor = np.array(marker_edit_anchor)

        dist_surface_edit = np.linalg.norm(pos - surface_edit_anchor)
        surface_edit_pressed = dist_surface_edit < 15

        dist_marker_edit = np.linalg.norm(pos - marker_edit_anchor)
        marker_edit_pressed = dist_marker_edit < 15

        return surface_edit_pressed, marker_edit_pressed

class State(Enum):
    SHOW_SURF = 1
    SHOW_IDS = 2
    SHOW_HEATMAP = 3