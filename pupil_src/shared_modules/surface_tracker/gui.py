"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import platform

import numpy as np
import cv2
import pyglui
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl
import glfw
import gl_utils

from surface_tracker import Heatmap_Mode


class GUI:
    """Custom GUI functionality for visualizing and editing surfaces"""

    def __init__(self, tracker):
        self.tracker = tracker

        self.heatmap_mode = Heatmap_Mode.WITHIN_SURFACE
        self.show_heatmap = False
        self.show_marker_ids = False

        self._edit_surf_corners = set()
        self._edit_surf_markers = set()
        self.heatmap_textures = {}
        self.surface_windows = {}

        self.color_primary = (1.0, 0.2, 0.6)
        self.color_secondary = (0.1, 1., 1.)
        self.color_tertiary = (0, 0.8, 0.7)

        self.glfont = pyglui.pyfontstash.fontstash.Context()
        self.glfont.add_font("opensans", pyglui.ui.get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

    def update(self):
        if self.show_heatmap:
            for surface in self.tracker.surfaces:
                if surface.detected:
                    self._draw_heatmap(surface)

        self._draw_markers()

        if self.show_marker_ids:
            for marker in self.tracker.markers:
                self._draw_marker_id(marker)

        for surface in self.tracker.surfaces:
            if not surface.detected:
                continue

            self._draw_surface_frames(surface)

            if surface in self._edit_surf_markers:
                self._draw_marker_toggles(surface)

            if surface in self._edit_surf_corners:
                self._draw_surface_corner_handles(surface)

            self.surface_windows[surface].update(self.tracker.g_pool.image_tex)

    def _draw_markers(self):
        color = pyglui_utils.RGBA(*self.color_secondary, .5)
        for m in self.tracker.markers:
            hat = np.array(
                [[[0, 0], [0, 1], [.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
            )
            hat = cv2.perspectiveTransform(hat, _get_norm_to_points_trans(m.verts))

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

        text = "conf: {:.2f}".format(marker.id_confidence)
        loc = anchor + (0, line_height * 2)
        self._draw_text(loc, text, self.color_secondary)

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
        frame = surface.map_from_surf(
            frame, self.tracker.camera_model, compensate_distortion=False
        )

        hat = np.array([[.3, .7], [.7, .7], [.5, .9], [.3, .7]], dtype=np.float32)
        hat = surface.map_from_surf(
            hat, self.tracker.camera_model, compensate_distortion=False
        )

        text_anchor = frame.reshape((5, -1))[2]
        text_anchor = text_anchor[0], text_anchor[1] - 75

        surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
        marker_edit_anchor = text_anchor[0], text_anchor[1] + 50

        return frame, hat, text_anchor, surface_edit_anchor, marker_edit_anchor

    def _draw_surf_menu(
        self, surface, text_anchor, surface_edit_anchor, marker_edit_anchor
    ):
        marker_detection_status = "{}   {}/{}".format(
            surface.name, surface.num_detected_markers, len(surface.reg_markers_dist)
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
                    [marker_edit_anchor], color=pyglui_utils.RGBA(*self.color_tertiary)
                )
            else:
                pyglui_utils.draw_points(
                    [marker_edit_anchor], color=pyglui_utils.RGBA(*self.color_primary)
                )

            if surface in self._edit_surf_corners:
                pyglui_utils.draw_points(
                    [surface_edit_anchor], color=pyglui_utils.RGBA(*self.color_tertiary)
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
            if marker.id in surface.reg_markers_dist.keys():
                active_markers.append(centroid)
            else:
                inactive_markers.append(centroid)

        pyglui_utils.draw_points(
            inactive_markers, size=20, color=pyglui_utils.RGBA(*self.color_primary, .8)
        )
        pyglui_utils.draw_points(
            active_markers, size=20, color=pyglui_utils.RGBA(*self.color_tertiary, .8)
        )

    def _draw_surface_corner_handles(self, surface):
        norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
        img_corners = surface.map_from_surf(
            norm_corners, self.tracker.camera_model, compensate_distortion=False
        )

        pyglui_utils.draw_points(
            img_corners, 20, pyglui_utils.RGBA(*self.color_primary, .5)
        )

    def _draw_heatmap(self, surface):
        if self.heatmap_mode == Heatmap_Mode.WITHIN_SURFACE:
            self.heatmap_textures[surface].update_from_ndarray(
                surface.within_surface_heatmap
            )
        else:
            self.heatmap_textures[surface].update_from_ndarray(
                surface.across_surface_heatmap
            )
        width, height = self.tracker.camera_model.resolution
        img_corners = np.array(
            [(0, height), (width, height), (width, 0), (0, 0)], dtype=np.float32
        )
        norm_trans = _get_points_to_norm_trans(img_corners)

        m = norm_trans @ surface.surf_to_dist_img_trans
        m = gl_utils.cvmat_to_glmat(m)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
        gl.glLoadMatrixf(m)
        self.heatmap_textures[surface].draw()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

    def on_click(self, pos, button, action):
        pos = np.array(pos, dtype=np.float32)

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
        norm_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
        for surface in self._edit_surf_corners:
            if surface.detected and surface.defined:
                img_corners = surface.map_from_surf(
                    norm_corners, self.tracker.camera_model, compensate_distortion=False
                )
                for idx, corner in enumerate(img_corners):
                    dist = np.linalg.norm(corner - pos)
                    if dist < 15:
                        if action == glfw.GLFW_PRESS:
                            self.tracker._edit_surf_verts.append((surface, idx))
                        elif action == glfw.GLFW_RELEASE:
                            self.tracker.notify_all(
                                {
                                    "subject": "surface_tracker.surfaces_changed",
                                    "uid": surface.uid,
                                }
                            )
                            self.tracker._edit_surf_verts = []

        # Marker Toggles
        if action == glfw.GLFW_PRESS:
            for surface in self._edit_surf_markers:
                if not surface.detected:
                    continue
                for marker in self.tracker.markers:
                    centroid = np.mean(marker.verts, axis=0)
                    dist = np.linalg.norm(centroid - pos)
                    if dist < 15:
                        if not marker.id in surface.reg_markers_dist.keys():
                            surface.add_marker(
                                marker.id, marker.verts, self.tracker.camera_model
                            )
                        else:
                            surface.pop_marker(marker.id)
                        self.tracker.notify_all(
                            {
                                "subject": "surface_tracker.surfaces_changed",
                                "uid": surface.uid,
                            }
                        )

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
        self.surface_windows[surface] = Surface_Window(surface, self.tracker)

    def remove_surface(self, surface):
        self.heatmap_textures.pop(surface)
        self._edit_surf_markers.discard(surface)
        self._edit_surf_corners.discard(surface)
        self.surface_windows.pop(surface)


def _get_norm_to_points_trans(points):
    norm_corners = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    return cv2.getPerspectiveTransform(norm_corners, np.array(points, dtype=np.float32))


def _get_points_to_norm_trans(points):
    norm_corners = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    return cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), norm_corners)


class Surface_Window:
    def __init__(self, surface, tracker):
        self.surface = surface
        self._window = None
        self.window_should_open = False
        self.window_should_close = False
        self.tracker = tracker

        # UI Platform tweaks
        if platform.system() == "Linux":
            self.window_position_default = (0, 0)
        elif platform.system() == "Windows":
            self.window_position_default = (8, 90)
        else:
            self.window_position_default = (0, 0)

    def open_close_window(self):
        if self._window:
            self.close_window()
        else:
            self.open_window()

    def open_window(self):
        if not self._window:

            monitor = None
            # open with same aspect ratio as surface
            height, width = (
                640,
                int(
                    640.
                    / (
                        self.surface.real_world_size["x"]
                        / self.surface.real_world_size["y"]
                    )
                ),
            )

            self._window = glfw.glfwCreateWindow(
                height,
                width,
                "Reference Surface: " + self.surface.name,
                monitor=monitor,
                share=glfw.glfwGetCurrentContext(),
            )

            glfw.glfwSetWindowPos(
                self._window,
                self.window_position_default[0],
                self.window_position_default[1],
            )

            self.trackball = gl_utils.trackball.Trackball()
            self.input = {"down": False, "mouse": (0, 0)}

            # Register callbacks
            glfw.glfwSetFramebufferSizeCallback(self._window, self.on_resize)
            glfw.glfwSetKeyCallback(self._window, self.on_window_key)
            glfw.glfwSetWindowCloseCallback(self._window, self.on_close)
            glfw.glfwSetMouseButtonCallback(self._window, self.on_window_mouse_button)
            glfw.glfwSetCursorPosCallback(self._window, self.on_pos)
            glfw.glfwSetScrollCallback(self._window, self.on_scroll)

            self.on_resize(self._window, *glfw.glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self._window)
            gl_utils.basic_gl_setup()
            gl_utils.make_coord_system_norm_based()

            # refresh speed settings
            glfw.glfwSwapInterval(0)

            glfw.glfwMakeContextCurrent(active_window)

    def close_window(self):
        if self._window:
            glfw.glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

    def update(self, tex):
        self.gl_display_in_window(tex)

    def gl_display_in_window(self, world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        if self._window and self.surface.detected:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self._window)
            gl_utils.clear_gl_screen()

            # cv uses 3x3 gl uses 4x4 tranformation matricies
            width, height = self.tracker.camera_model.resolution
            img_corners = np.array(
                [(0, height), (width, height), (width, 0), (0, 0)], dtype=np.float32
            )
            denorm_trans = _get_norm_to_points_trans(img_corners)

            m = self.surface.dist_img_to_surf_trans @ denorm_trans
            m = gl_utils.cvmat_to_glmat(m)

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPushMatrix()
            gl.glLoadIdentity()
            gl.glOrtho(0, 1, 0, 1, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
            # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            gl.glLoadMatrixf(m)

            world_tex.draw()

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPopMatrix()

            # # Draw recent pupil positions onto the surface:
            try:
                for gp in self.surface.gaze_history:
                    pyglui_utils.draw_points(
                        [gp["gaze"]],
                        color=pyglui_utils.RGBA(0.0, 0.8, 0.5, 0.8),
                        size=80,
                    )
            except AttributeError:
                # If gaze_history does not exist, we are in the Surface_Tracker_Offline.
                # In this case gaze visualizations will be drawn directly onto the scene
                # image and thus propagate to the surface crop automatically.
                pass

            glfw.glfwSwapBuffers(self._window)
            glfw.glfwMakeContextCurrent(active_window)

    def on_resize(self, window, w, h):
        self.trackball.set_window_size(w, h)
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.glfwMakeContextCurrent(active_window)

    def on_window_key(self, window, key, scancode, action, mods):
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_ESCAPE:
                self.on_close()

    def on_close(self, window=None):
        self.close_window()

    def on_window_mouse_button(self, window, button, action, mods):
        if action == glfw.GLFW_PRESS:
            self.input["down"] = True
            self.input["mouse"] = glfw.glfwGetCursorPos(window)
        if action == glfw.GLFW_RELEASE:
            self.input["down"] = False

    def on_pos(self, window, x, y):
        if self.input["down"]:
            old_x, old_y = self.input["mouse"]
            self.trackball.drag_to(x - old_x, y - old_y)
            self.input["mouse"] = x, y

    def on_scroll(self, window, x, y):
        self.trackball.zoom_to(y)
