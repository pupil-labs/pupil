from enum import Enum

import numpy as np
import cv2
import pyglui
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl
import glfw
import gl_utils

from .surface import _Surface_Marker


class GUI:
    """Custom GUI functionality for visualizing and editing surfaces"""

    def __init__(self, tracker):
        self.tracker = tracker

        self.state = State.SHOW_SURF
        self._edit_surf_corners = set()
        self._edit_surf_markers = set()

        self.color_primary = (1.0, 0.2, 0.6)
        self.color_secondary = (0.1, 1., 1.)
        self.color_highlight = (0, 0.8, 0.7)

        self.glfont = pyglui.pyfontstash.fontstash.Context()
        self.glfont.add_font("opensans", pyglui.ui.get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

        self.heatmap_textures = {}

    def update(self):
        self._draw_markers()

        if self.state == State.SHOW_IDS:
            for marker in self.tracker.markers:
                self._draw_marker_id(marker)

        for surface in self.tracker.surfaces:
            if not surface.detected:
                continue

            if self.state == State.SHOW_SURF:
                self._draw_surface_frames(surface)

                if surface in self._edit_surf_markers:
                    self._draw_marker_toggles(surface)

                if surface in self._edit_surf_corners:
                    self._draw_surface_corner_handles(surface)

            elif self.state == State.SHOW_HEATMAP:
                self.heatmap_textures[surface].update_from_ndarray(surface.heatmap)

                m = gl_utils.cvmat_to_glmat(surface._surf_to_dist_img_trans)

                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glPushMatrix()
                gl.glLoadIdentity()
                gl.glOrtho(
                    0, self.tracker.camera_model.resolution[0], self.tracker.camera_model.resolution[1], 0, -1, 1
                )

                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glPushMatrix()
                # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
                gl.glLoadMatrixf(m)

                self.heatmap_textures[surface_idx].draw()

                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glPopMatrix()
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glPopMatrix()

            #
            # for s in self.surfaces:
            #     if self.locate_3d:
            #         s.gl_display_in_window_3d(self.g_pool.image_tex)
            #     else:
            #         s.gl_display_in_window(self.g_pool.image_tex)

    def _draw_markers(self):
        color = pyglui_utils.RGBA(*self.color_secondary, .5)
        for m in self.tracker.markers:
            # TODO Update to new marker class
            # TODO Marker.verts has shape (N,1,2), change to (N,2)
            hat = np.array(
                [[[0, 0], [0, 1], [.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
            )
            hat = cv2.perspectiveTransform(hat, self._get_marker_to_img_trans(m))

            if (
                m.perimeter >= self.tracker.marker_min_perimeter
                and m.id_confidence > self.tracker.marker_min_confidence
            ):
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color)
                pyglui_utils.draw_polyline(
                    hat.reshape((6, 2)), color=color, line_type=gl.GL_POLYGON
                )
            else:
                pyglui_utils.draw_polyline(hat.reshape((6, 2)), color=color)

    def _draw_marker_id(self, marker):
        verts = np.array(marker.verts, dtype=np.float32)
        verts.shape = (4, 2)
        anchor = np.array([np.min(verts[:, 0]), np.max(verts[:, 1])])
        line_height = 16

        text = "id: {:d}".format(marker.id)
        loc = anchor + (0, line_height * 1)
        self._draw_text(loc, text, self.color_secondary)

        text = "conf: {:.3f}".format(marker.id_confidence)
        loc = anchor + (0, line_height * 2)
        self._draw_text(loc, text, self.color_secondary)

    def _get_marker_to_img_trans(self, marker):
        norm_corners = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
        return cv2.getPerspectiveTransform(
            norm_corners, np.array(marker.verts, dtype=np.float32)
        )

    def _draw_surface_frames(self, surface):
        if not surface.detected:
            return

        frame, hat, text_anchor, surface_edit_anchor, marker_edit_anchor = self._get_surface_anchor_points(
            surface
        )
        alpha = min(1, surface.build_up_status)

        pyglui_utils.draw_polyline(
            frame.reshape((5, 2)),
            1,
            color=pyglui_utils.RGBA(*self.color_primary, alpha),
        )
        pyglui_utils.draw_polyline(
            hat.reshape((4, 2)), 1, color=pyglui_utils.RGBA(*self.color_primary, alpha)
        )

        self._draw_surf_menu(
            surface, text_anchor, surface_edit_anchor, marker_edit_anchor
        )

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

    def _draw_surf_menu(
        self, surface, text_anchor, surface_edit_anchor, marker_edit_anchor
    ):
        marker_detection_status = "{}   {}/{}".format(
            surface.name, surface.num_detected_markers, len(surface.reg_markers)
        )
        self._draw_text(
            (text_anchor[0] + 15, text_anchor[1] + 6),
            marker_detection_status,
            self.color_secondary,
        )

        # If the surface is defined, draw menu buttons. Otherwise draw definition
        # progress.
        if surface.defined:

            # Buttons
            if surface in self._edit_surf_markers:
                pyglui_utils.draw_points(
                    [marker_edit_anchor], color=pyglui_utils.RGBA(*self.color_highlight)
                )
            else:
                pyglui_utils.draw_points(
                    [marker_edit_anchor], color=pyglui_utils.RGBA(*self.color_primary)
                )

            if surface in self._edit_surf_corners:
                pyglui_utils.draw_points(
                    [surface_edit_anchor],
                    color=pyglui_utils.RGBA(*self.color_highlight),
                )
            else:
                pyglui_utils.draw_points(
                    [surface_edit_anchor], color=pyglui_utils.RGBA(*self.color_primary)
                )

            # Text
            self._draw_text(
                (surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6),
                "edit surface",
                self.color_secondary,
            )
            self._draw_text(
                (marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6),
                "add/remove markers",
                self.color_secondary,
            )
        else:
            progress_text = "{:.0f} %".format(surface.build_up_status * 100)

            self._draw_text(
                (surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6),
                "Learning affiliated markers...",
                self.color_secondary,
            )
            self._draw_text(
                (marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6),
                progress_text,
                self.color_secondary,
            )

    def _draw_text(self, loc, text, color):
        self.glfont.set_blur(3.9)
        self.glfont.set_color_float((0, 0, 0, .8))
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

            centroid = np.mean(marker.verts, axis=0)
            centroid = (centroid[0, 0], centroid[0, 1])
            if marker.id in surface.reg_markers.keys():
                active_markers.append(centroid)
            else:
                inactive_markers.append(centroid)

        pyglui_utils.draw_points(
            inactive_markers, size=20, color=pyglui_utils.RGBA(*self.color_primary, .8)
        )
        pyglui_utils.draw_points(
            active_markers, size=20, color=pyglui_utils.RGBA(*self.color_highlight, .8)
        )

    def _draw_surface_corner_handles(self, surface):
        norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
        img_corners = surface.map_from_surf(norm_corners)

        pyglui_utils.draw_points(
            img_corners, 20, pyglui_utils.RGBA(*self.color_primary, .5)
        )

    def on_click(self, pos, button, action):
        pos = np.array(pos, dtype=np.float32)

        if self.state == State.SHOW_SURF:

            # Menu Buttons
            if action == glfw.GLFW_PRESS:
                for surface in self.tracker.surfaces:

                    if not surface.detected:
                        continue

                    surface_edit_pressed, marker_edit_pressed = self._check_surface_button_pressed(
                        surface, pos
                    )

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
                norm_corners = np.array(
                    [(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32
                )

                for surface in self._edit_surf_corners:
                    if surface.detected and surface.defined:
                        img_corners = surface.map_from_surf(norm_corners)
                        for idx, corner in enumerate(img_corners):
                            dist = np.linalg.norm(corner - pos)
                            if dist < 15:
                                self.tracker._edit_surf_verts.append((surface, idx))
            elif action == glfw.GLFW_RELEASE:
                if self.tracker._edit_surf_verts:
                    self.tracker.notify_all({"subject": "surfaces_changed"})
                self.tracker._edit_surf_verts = []



            # Marker Toggles
            if action == glfw.GLFW_PRESS:
                for surface in self._edit_surf_markers:
                    for marker in self.tracker.markers:
                        centroid = np.mean(marker.verts, axis=0)
                        dist = np.linalg.norm(centroid - pos)
                        if dist < 15:
                            if not marker.id in surface.reg_markers.keys():
                                surface_marker = _Surface_Marker(marker.id)
                                marker_verts = np.array(marker.verts).reshape((4, 2))
                                uv_coords = surface.map_to_surf(marker_verts)
                                surface_marker.add_observation(uv_coords)
                                surface.reg_markers[marker.id] = surface_marker
                            else:
                                surface.reg_markers.pop(marker.id)
                            self.tracker.notify_all({"subject": "surfaces_changed"})

    def _check_surface_button_pressed(self, surface, pos):
        frame, hat, text_anchor, surface_edit_anchor, marker_edit_anchor = self._get_surface_anchor_points(
            surface
        )

        surface_edit_anchor = np.array(surface_edit_anchor)
        marker_edit_anchor = np.array(marker_edit_anchor)

        dist_surface_edit = np.linalg.norm(pos - surface_edit_anchor)
        surface_edit_pressed = dist_surface_edit < 15

        dist_marker_edit = np.linalg.norm(pos - marker_edit_anchor)
        marker_edit_pressed = dist_marker_edit < 15

        return surface_edit_pressed, marker_edit_pressed

    def add_surface(self, surface):
        self.heatmap_textures[surface] = pyglui_utils.Named_Texture()

    def remove_surface(self, surface):
        self.heatmap_textures.pop(surface)
        self._edit_surf_markers.remove(surface)
        self._edit_surf_corners.remove(surface)

class State(Enum):
    SHOW_SURF = "Show Markers and Surfaces"
    SHOW_IDS = "Show marker IDs"
    SHOW_HEATMAP = "Show Heatmaps"
