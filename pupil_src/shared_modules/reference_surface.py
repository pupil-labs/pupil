"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from platform import system
from time import time
from collections import deque
import random

import numpy as np
import cv2

import glfw
from OpenGL.GL import *
from OpenGL.GL import GL_LINES

from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from pyglui.cygl.utils import (
    RGBA,
    draw_polyline_norm,
    draw_polyline,
    draw_points_norm,
    draw_points,
    Named_Texture,
)

from gl_utils import (
    adjust_gl_view,
    clear_gl_screen,
    basic_gl_setup,
    cvmat_to_glmat,
    make_coord_system_norm_based,
)
from gl_utils.trackball import Trackball

from methods import GetAnglesPolyline, normalize, denormalize

import logging

logger = logging.getLogger(__name__)
surface_corners_norm = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)


def m_verts_to_screen(verts):
    # verts need to be sorted counter-clockwise stating at bottom left
    return cv2.getPerspectiveTransform(surface_corners_norm, verts)


def m_verts_from_screen(verts):
    # verts need to be sorted counter-clockwise stating at bottom left
    return cv2.getPerspectiveTransform(verts, surface_corners_norm)


class Reference_Surface(object):
    """docstring for Reference Surface

    The surface coodinate system is 0-1.
    Origin is the bottom left corner, (1,1) is the top right

    The first scalar in the pos vector is width we call this 'u'.
    The second is height we call this 'v'.
    The surface is thus defined by 4 vertecies:
        Our convention is this order: (0,0),(1,0),(1,1),(0,1)

    The surface is supported by a set of n>=1 Markers:
        Each marker has an id, you can not not have markers with the same id twice.
        Each marker has 4 verts (order is the same as the surface verts)
        Each maker vertex has a uv coord that places it on the surface

    When we find the surface in locate() we use the correspondence
    of uv and screen coords of all 4 verts of all detected markers to get the
    surface to screen homography.

    This allows us to get homographies for partially visible surfaces,
    all we need are 2 visible markers. (We could get away with just
    one marker but in pracise this is to noisy.)
    The more markers we find the more accurate the homography.

    """

    def __init__(self, g_pool, name="unnamed", saved_definition=None):
        self.g_pool = g_pool
        self.camera_model = g_pool.capture.intrinsics
        self.name = name
        self.markers = {}
        self.detected_markers = 0
        self.defined = False
        self.build_up_status = 0
        self.required_build_up = 90.
        self.detected = False
        self.m_surface_to_img = None
        self.m_img_to_surface = None
        self.m_surface_to_img_distorted = None
        self.m_img_to_surface_distorted = None
        self.camera_pose_3d = None
        self.use_distortion = True

        self.uid = "{:04}".format(random.randint(0, 9999))
        self.real_world_size = {"x": 1., "y": 1.}

        self.heatmap = np.ones(0)
        self.heatmap_detail = .2
        self.heatmap_texture = Named_Texture()
        self.gaze_history = deque()
        self.gaze_history_length = 1.0  # unit: seconds

        # window and gui vars
        self._window = None
        self.fullscreen = False
        self.window_should_open = False
        self.window_should_close = False

        self.gaze_on_srf = []  # points on surface for realtime feedback display
        self.fixations_on_srf = []  # fixations on surface

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

        self.old_corners_robust = None
        if saved_definition is not None:
            self.load_from_dict(saved_definition)

        # UI Platform tweaks
        if system() == "Linux":
            self.window_position_default = (0, 0)
        elif system() == "Windows":
            self.window_position_default = (8, 90)
        else:
            self.window_position_default = (0, 0)

    def save_to_dict(self):
        """
        save all markers and name of this surface to a dict.
        """
        markers = dict(
            [(m_id, m.uv_coords.tolist()) for m_id, m in self.markers.items()]
        )
        return {
            "name": self.name,
            "uid": self.uid,
            "markers": markers,
            "real_world_size": self.real_world_size,
            "gaze_history_length": self.gaze_history_length,
        }

    def load_from_dict(self, d):
        """
        load all markers of this surface to a dict.
        """
        self.name = d["name"]
        self.uid = d["uid"]
        self.gaze_history_length = d.get(
            "gaze_history_length", self.gaze_history_length
        )
        self.real_world_size = d.get("real_world_size", {"x": 1., "y": 1.})

        marker_dict = d["markers"]
        for m_id, uv_coords in marker_dict.items():
            self.markers[m_id] = Support_Marker(m_id)
            self.markers[m_id].load_uv_coords(np.asarray(uv_coords))

        # flag this surface as fully defined
        self.defined = True
        self.build_up_status = self.required_build_up

    def build_correspondence(
        self, visible_markers, min_marker_perimeter, min_id_confidence
    ):
        """
        - use all visible markers
        - fit a convex quadrangle around it
        - use quadrangle verts to establish perpective transform
        - map all markers into surface space
        - build up list of found markers and their uv coords
        """
        usable_markers = [
            m for m in visible_markers if m["perimeter"] >= min_marker_perimeter
        ]
        all_verts = [m["verts"] for m in usable_markers]
        if not all_verts:
            return
        all_verts = np.array(all_verts, dtype=np.float32)
        all_verts.shape = (
            -1,
            1,
            2,
        )  # [vert,vert,vert,vert,vert...] with vert = [[r,c]]

        all_verts_undistorted_normalized = self.g_pool.capture.intrinsics.unprojectPoints(
            all_verts, use_distortion=self.use_distortion
        )[
            :, :2
        ]  # we ommit the z corrd as it is 1.
        all_verts_undistorted_normalized.shape = -1, 1, 2
        hull = cv2.convexHull(
            all_verts_undistorted_normalized.astype(np.float32), clockwise=False
        )

        # simplify until we have excatly 4 verts
        if hull.shape[0] > 4:
            new_hull = cv2.approxPolyDP(hull, epsilon=1, closed=True)
            if new_hull.shape[0] >= 4:
                hull = new_hull
        if hull.shape[0] > 4:
            curvature = abs(GetAnglesPolyline(hull, closed=True))
            most_acute_4_threshold = sorted(curvature)[3]
            hull = hull[curvature <= most_acute_4_threshold]

        # all_verts_undistorted_normalized space is flipped in y.
        # we need to change the order of the hull vertecies
        hull = hull[[1, 0, 3, 2], :, :]

        # now we need to roll the hull verts until we have the right orientation:
        # all_verts_undistorted_normalized space has its origin at the image center.
        # adding 1 to the coordinates puts the origin at the top left.
        distance_to_top_left = np.sqrt(
            (hull[:, :, 0] + 1) ** 2 + (hull[:, :, 1] + 1) ** 2
        )
        bot_left_idx = np.argmin(distance_to_top_left) + 1
        hull = np.roll(hull, -bot_left_idx, axis=0)

        # based on these 4 verts we calculate the transformations into a 0,0 1,1 square space
        m_from_undistored_norm_space = m_verts_from_screen(hull)
        self.detected = True
        # map the markers vertices into the surface space (one can think of these as texture coordinates u,v)
        marker_uv_coords = cv2.perspectiveTransform(
            all_verts_undistorted_normalized, m_from_undistored_norm_space
        )
        marker_uv_coords.shape = (
            -1,
            4,
            1,
            2,
        )  # [marker,marker...] marker = [ [[r,c]],[[r,c]] ]

        # build up a dict of discovered markers. Each with a history of uv coordinates
        for m, uv in zip(usable_markers, marker_uv_coords):
            try:
                self.markers[m["id"]].add_uv_coords(uv)
            except KeyError:
                self.markers[m["id"]] = Support_Marker(m["id"])
                self.markers[m["id"]].add_uv_coords(uv)

        # average collection of uv correspondences across detected markers
        self.build_up_status = sum(
            [len(m.collected_uv_coords) for m in self.markers.values()]
        ) / float(len(self.markers))

        if self.build_up_status >= self.required_build_up:
            self.finalize_correspondence()

    def finalize_correspondence(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        for k, m in self.markers.items():
            if len(m.collected_uv_coords) > self.required_build_up * .5:
                persistent_markers[k] = m
        self.markers = persistent_markers
        for m in self.markers.values():
            m.compute_robust_mean()

        self.defined = True
        if hasattr(self, "on_finish_define"):
            self.on_finish_define()
            del self.on_finish_define

    def update_gaze_history(self):
        self.gaze_history.extend(self.gaze_on_srf)
        try:  # use newest gaze point to determine age threshold
            age_threshold = (
                self.gaze_history[-1]["timestamp"] - self.gaze_history_length
            )
            while self.gaze_history[1]["timestamp"] < age_threshold:
                self.gaze_history.popleft()  # remove outdated gaze points
        except IndexError:
            pass

    def generate_heatmap(self):
        data = [gp["norm_pos"] for gp in self.gaze_history if gp["confidence"] > 0.6]
        self._generate_heatmap(data)

    def _generate_heatmap(self, data):
        if not data:
            return

        grid = int(self.real_world_size["y"]), int(self.real_world_size["x"])

        xvals, yvals = zip(*((x, 1. - y) for x, y in data))
        hist, *edges = np.histogram2d(
            yvals, xvals, bins=grid, range=[[0, 1.], [0, 1.]], normed=False
        )
        filter_h = int(self.heatmap_detail * grid[0]) // 2 * 2 + 1
        filter_w = int(self.heatmap_detail * grid[1]) // 2 * 2 + 1
        hist = cv2.GaussianBlur(hist, (filter_h, filter_w), 0)

        hist_max = hist.max()
        hist *= (255. / hist_max) if hist_max else 0.
        hist = hist.astype(np.uint8)
        c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
        # reuse allocated memory if possible
        if self.heatmap.shape != (*grid, 4):
            self.heatmap = np.ones((*grid, 4), dtype=np.uint8)
            self.heatmap[:, :, 3] = 125
        self.heatmap[:, :, :3] = c_map
        self.heatmap_texture.update_from_ndarray(self.heatmap)

    def gl_display_heatmap(self):
        if self.detected:
            m = cvmat_to_glmat(self.m_surface_to_img_distorted)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.camera_model.resolution[0],
                    self.camera_model.resolution[1], 0, -1, 1)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.heatmap_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def locate(
        self, visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False
    ):
        """
        - find overlapping set of surface markers and visible_markers
        - compute homography (and inverse) based on this subset
        """

        if not self.defined:
            self.build_correspondence(
                visible_markers, min_marker_perimeter, min_id_confidence
            )

        res = self._get_location(
            visible_markers, min_marker_perimeter, min_id_confidence, locate_3d
        )

    def _get_location(
        self, visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False
    ):

        visible_markers_filtered = self._filter_markers(
            visible_markers, min_id_confidence, min_marker_perimeter
        )
        visible_registered_markers = set(visible_markers_filtered.keys()) & set(
            self.markers.keys()
        )

        if not visible_registered_markers or len(visible_registered_markers) < min(
            2, len(self.markers)
        ):
            detected = False
        else:
            visible_verts = np.array(
                [
                    visible_markers_filtered[i]["verts"]
                    for i in visible_registered_markers
                ]
            )
            registered_verts = np.array(
                [self.markers[i].uv_coords for i in visible_registered_markers]
            )

            # [vert,vert,vert,vert,vert...] with vert = [[r,c]]
            visible_verts.shape = (-1, 1, 2)
            registered_verts.shape = (-1, 1, 2)

            m_img_to_surface_distorted, m_surface_to_img_distorted = \
                self._findHomographies(
                registered_verts, visible_verts
            )

            visible_verts_undistorted = self.camera_model.undistortPoints(visible_verts)
            # visible_verts_undistorted = visible_verts

            m_img_to_surface, m_surface_to_img = self._findHomographies(
                registered_verts, visible_verts_undistorted
            )


            detected = not m_img_to_surface is None


        if detected == False:
            m_img_to_surface = None
            m_surface_to_img = None
            m_img_to_surface_distorted = None
            m_surface_to_img_distorted = None

        self.detected = detected
        self.detected_markers = len(visible_registered_markers),
        self.m_surface_to_img = m_surface_to_img
        self.m_img_to_surface = m_img_to_surface
        self.m_surface_to_img_distorted = m_surface_to_img_distorted
        self.m_img_to_surface_distorted = m_img_to_surface_distorted
        self.camera_pose_3d = None


    def _filter_markers(self, visible_markers, min_id_confidence, min_marker_perimeter):
        filtered_markers = [
            m
            for m in visible_markers
            if m["perimeter"] >= min_marker_perimeter
            and m["id_confidence"] > min_id_confidence
        ]

        # if an id shows twice use the bigger marker (usually this is a screen camera echo artifact.)
        marker_by_id = {}
        for m in filtered_markers:
            if (
                not m["id"] in marker_by_id
                or m["perimeter"] > marker_by_id[m["id"]]["perimeter"]
            ):
                marker_by_id[m["id"]] = m

        return marker_by_id

    def _findHomographies(self, points_A, points_B):
        points_B.shape = -1, 1, 2
        points_A.shape = -1, 1, 2

        B_to_A, mask = cv2.findHomography(
            points_A, points_B, method=cv2.RANSAC, ransacReprojThreshold=100
        )

        if not mask.all():
            return None, None

        A_to_B, mask = cv2.findHomography(points_B, points_A)
        return A_to_B, B_to_A

    def img_to_surface(self, points):
        shape = points.shape
        points = self.camera_model.undistortPoints(points)
        points.shape = (-1, 1, 2)
        new_pos = cv2.perspectiveTransform(points, self.m_img_to_surface)
        new_pos.shape = shape

        return new_pos

    def surface_to_img(self, points):
        shape = points.shape
        points.shape = (-1, 1, 2)
        new_pos = cv2.perspectiveTransform(points, self.m_surface_to_img)
        new_pos = self.camera_model.distortPoints(new_pos)
        new_pos.shape = shape

        return new_pos

    def map_gaze_to_surface(self, gaze_data):
        if not gaze_data:
            return gaze_data

        gaze_points = np.array([denormalize(g["norm_pos"],
                                            self.g_pool.capture.intrinsics.resolution, flip_y=True)
        for g in
                                gaze_data])
        gaze_on_surface = self.img_to_surface(gaze_points)

        result = []
        for sample, gof in zip(gaze_data, gaze_on_surface):
            on_srf = bool((0 <= gof[0] <= 1) and (0 <= gof[1] <= 1))
            result.append({
                "topic": sample["topic"] + "_on_surface",
                "norm_pos": gof.tolist(),
                "confidence": sample["confidence"],
                "on_srf": on_srf,
                "base_data": sample,
                "timestamp": sample["timestamp"],
            })
        return result

    def map_data_to_surface(self, data, m_from_screen):
        return [self.map_datum_to_surface(d, m_from_screen) for d in data]

    def move_vertex(self, vert_idx, new_pos_img):
        """
        this fn is used to manipulate the surface boundary (coordinate system)
        new_pos is in uv-space coords
        if we move one vertex of the surface we need to find
        the tranformation from old quadrangle to new quardangle
        and apply that transformation to our marker uv-coords
        """

        # new_pos_img = self.camera_model.undistortPoints([new_pos_img])
        new_pos_surface = self.img_to_surface(new_pos_img)
        before = surface_corners_norm
        after = before.copy()
        after[vert_idx] = new_pos_surface
        transform = cv2.getPerspectiveTransform(after, before)
        for m in self.markers.values():
            m.uv_coords = cv2.perspectiveTransform(m.uv_coords, transform)

    def add_marker(
        self, marker, visible_markers, min_marker_perimeter, min_id_confidence
    ):
        """
        add marker to surface.
        """
        res = self._get_location(
            visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False
        )
        if res["detected"]:
            support_marker = Support_Marker(marker["id"])
            marker_verts = np.array(marker["verts"])
            marker_verts_undistorted_normalized = self.g_pool.capture.intrinsics.unprojectPoints(
                marker_verts, use_distortion=self.use_distortion
            )[
                :, :2
            ]
            marker_verts_undistorted_normalized.shape = -1, 1, 2
            marker_uv_coords = cv2.perspectiveTransform(
                marker_verts_undistorted_normalized, res["m_from_undistored_norm_space"]
            )
            support_marker.load_uv_coords(marker_uv_coords)
            self.markers[marker["id"]] = support_marker

    def remove_marker(self, marker):
        if len(self.markers) == 1:
            logger.warning(
                "Need at least one marker per surface. Will not remove this last marker."
            )
            return
        self.markers.pop(marker["id"])

    def marker_status(self):
        return "{}   {}/{}".format(self.name, self.detected_markers, len(self.markers))

    def get_mode_toggle(self, pos, img_shape):
        if self.detected and self.defined:
            x, y = pos
            frame = np.array(
                [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]], dtype=np.float32
            )
            frame = cv2.perspectiveTransform(frame, self.m_surface_to_img)
            frame = self.camera_model.distortPoints(frame)

            text_anchor = frame.reshape((5, -1))[2]
            text_anchor = text_anchor[0], text_anchor[1] - 75
            surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
            marker_edit_anchor = text_anchor[0], text_anchor[1] + 50
            if (
                np.sqrt(
                    (x - surface_edit_anchor[0]) ** 2
                    + (y - surface_edit_anchor[1]) ** 2
                )
                < 15
            ):
                return "surface_mode"
            elif (
                np.sqrt(
                    (x - marker_edit_anchor[0]) ** 2 + (y - marker_edit_anchor[1]) ** 2
                )
                < 15
            ):
                return "marker_mode"
            else:
                return None
        else:
            return None

    def gl_draw_frame(
        self,
        img_size,
        color=(1.0, 0.2, 0.6, 1.0),
        highlight=False,
        surface_mode=False,
        marker_mode=False,
    ):
        """
        draw surface and markers
        """
        if self.detected:
            r, g, b, a = color
            frame = np.array(
                [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]], dtype=np.float32
            )
            hat = np.array([[[.3, .7], [.7, .7], [.5, .9], [.3, .7]]], dtype=np.float32)
            hat = cv2.perspectiveTransform(hat, self.m_surface_to_img)
            hat = self.camera_model.distortPoints(hat)
            frame = cv2.perspectiveTransform(frame, self.m_surface_to_img)
            frame = self.camera_model.distortPoints(frame)
            alpha = min(1, self.build_up_status / self.required_build_up)
            if highlight:
                draw_polyline_norm(
                    frame.reshape((5, 2)),
                    1,
                    RGBA(r, g, b, a * .1),
                    line_type=GL_POLYGON,
                )
            draw_polyline(frame.reshape((5, 2)), 1, RGBA(r, g, b, a * alpha))
            draw_polyline(hat.reshape((4, 2)), 1, RGBA(r, g, b, a * alpha))
            text_anchor = frame.reshape((5, -1))[2]
            text_anchor = text_anchor[0], text_anchor[1] - 75
            surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
            marker_edit_anchor = text_anchor[0], text_anchor[1] + 50
            if self.defined:
                if marker_mode:
                    draw_points([marker_edit_anchor], color=RGBA(0, .8, .7))
                else:
                    draw_points([marker_edit_anchor])
                if surface_mode:
                    draw_points([surface_edit_anchor], color=RGBA(0, .8, .7))
                else:
                    draw_points([surface_edit_anchor])

                self.glfont.set_blur(3.9)
                self.glfont.set_color_float((0, 0, 0, .8))
                self.glfont.draw_text(
                    text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
                )
                self.glfont.draw_text(
                    surface_edit_anchor[0] + 15,
                    surface_edit_anchor[1] + 6,
                    "edit surface",
                )
                self.glfont.draw_text(
                    marker_edit_anchor[0] + 15,
                    marker_edit_anchor[1] + 6,
                    "add/remove markers",
                )
                self.glfont.set_blur(0.0)
                self.glfont.set_color_float((0.1, 8., 8., .9))
                self.glfont.draw_text(
                    text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
                )
                self.glfont.draw_text(
                    surface_edit_anchor[0] + 15,
                    surface_edit_anchor[1] + 6,
                    "edit surface",
                )
                self.glfont.draw_text(
                    marker_edit_anchor[0] + 15,
                    marker_edit_anchor[1] + 6,
                    "add/remove markers",
                )
            else:
                progress = (self.build_up_status / float(self.required_build_up)) * 100
                progress_text = "%.0f%%" % progress
                self.glfont.set_blur(3.9)
                self.glfont.set_color_float((0, 0, 0, .8))
                self.glfont.draw_text(
                    text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
                )
                self.glfont.draw_text(
                    surface_edit_anchor[0] + 15,
                    surface_edit_anchor[1] + 6,
                    "Learning affiliated markers...",
                )
                self.glfont.draw_text(
                    marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6, progress_text
                )
                self.glfont.set_blur(0.0)
                self.glfont.set_color_float((0.1, 8., 8., .9))
                self.glfont.draw_text(
                    text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status()
                )
                self.glfont.draw_text(
                    surface_edit_anchor[0] + 15,
                    surface_edit_anchor[1] + 6,
                    "Learning affiliated markers...",
                )
                self.glfont.draw_text(
                    marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6, progress_text
                )

    def gl_draw_corners(self):
        """
        draw surface and markers
        """
        if self.detected:
            frame = cv2.perspectiveTransform(
                surface_corners_norm.reshape(-1, 1, 2), self.m_surface_to_img
            )
            frame = self.camera_model.distortPoints(frame)
            draw_points(frame.reshape((4, 2)), 20, RGBA(1.0, 0.2, 0.6, .5))

    #### fns to draw surface in seperate window
    def gl_display_in_window(self, world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        if self._window and self.detected:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self._window)
            clear_gl_screen()

            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_img_to_surface)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1, -1, 1)  # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            world_tex.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            # now lets get recent pupil positions on this surface:
            for gp in self.gaze_on_srf:
                draw_points_norm(
                    [gp["norm_pos"]], color=RGBA(0.0, 0.8, 0.5, 0.8), size=80
                )

            glfw.glfwSwapBuffers(self._window)
            glfw.glfwMakeContextCurrent(active_window)

    #### fns to draw surface in separate window
    def gl_display_in_window_3d(self, world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        K, img_size = (
            self.g_pool.capture.intrinsics.K,
            self.g_pool.capture.intrinsics.resolution,
        )

        if self._window and self.camera_pose_3d is not None:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self._window)
            glClearColor(.8, .8, .8, 1.)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearDepth(1.0)
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_TEST)
            self.trackball.push()

            glMatrixMode(GL_MODELVIEW)

            draw_coordinate_system(l=self.real_world_size["x"])
            glPushMatrix()
            glScalef(self.real_world_size["x"], self.real_world_size["y"], 1)
            draw_polyline(
                [[0, 0], [0, 1], [1, 1], [1, 0]],
                color=RGBA(.5, .3, .1, .5),
                thickness=3,
            )
            glPopMatrix()
            # Draw the world window as projected onto the plane using the homography mapping
            glPushMatrix()
            glScalef(self.real_world_size["x"], self.real_world_size["y"], 1)
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_img_to_surface)
            glMultMatrixf(m)
            glTranslatef(0, 0, -.01)
            world_tex.draw()
            draw_polyline(
                [[0, 0], [0, 1], [1, 1], [1, 0]],
                color=RGBA(.5, .3, .6, .5),
                thickness=3,
            )
            glPopMatrix()

            # Draw the camera frustum and origin using the 3d tranformation obtained from solvepnp
            glPushMatrix()
            glMultMatrixf(self.camera_pose_3d.T.flatten())
            draw_frustum(img_size, K, 150)
            glLineWidth(1)
            draw_frustum(img_size, K, .1)
            draw_coordinate_system(l=5)
            glPopMatrix()

            self.trackball.pop()

            glfw.glfwSwapBuffers(self._window)
            glfw.glfwMakeContextCurrent(active_window)

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                monitor = glfw.glfwGetMonitors()[self.monitor_idx]
                mode = glfw.glfwGetVideoMode(monitor)
                height, width = mode[0], mode[1]
            else:
                monitor = None
                height, width = (
                    640,
                    int(640. / (self.real_world_size["x"] / self.real_world_size["y"])),
                )  # open with same aspect ratio as surface

            self._window = glfw.glfwCreateWindow(
                height,
                width,
                "Reference Surface: " + self.name,
                monitor=monitor,
                share=glfw.glfwGetCurrentContext(),
            )
            if not self.fullscreen:
                glfw.glfwSetWindowPos(
                    self._window,
                    self.window_position_default[0],
                    self.window_position_default[1],
                )

            self.trackball = Trackball()
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
            basic_gl_setup()
            make_coord_system_norm_based()

            # refresh speed settings
            glfw.glfwSwapInterval(0)

            glfw.glfwMakeContextCurrent(active_window)

    def close_window(self):
        if self._window:
            glfw.glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

    def open_close_window(self):
        if self._window:
            self.close_window()
        else:
            self.open_window()

    # window calbacks
    def on_resize(self, window, w, h):
        self.trackball.set_window_size(w, h)
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        adjust_gl_view(w, h)
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

    def cleanup(self):
        if self._window:
            self.close_window()


class Support_Marker(object):
    """
    This is a class only to be used by Reference_Surface
    it decribes the used markers with the uv coords of its verts.
    """

    def __init__(self, uid):
        self.uid = uid
        self.uv_coords = None
        self.collected_uv_coords = []
        self.robust_uv_cords = False

    def load_uv_coords(self, uv_coords):
        self.uv_coords = uv_coords
        self.robust_uv_cords = True

    def add_uv_coords(self, uv_coords):
        self.collected_uv_coords.append(uv_coords)
        self.uv_coords = uv_coords

    def compute_robust_mean(self, threshhold=.1):
        """
        treat 50% as outliers. Assume majory is right.
        """
        # a stacked list of marker uv coords. marker uv cords are 4 verts with each a uv position.
        uv = np.array(self.collected_uv_coords)
        # # the mean marker uv_coords including outliers
        base_line_mean = np.mean(uv, axis=0)
        # # devidation is the distance of each scalar (4*2 per marker to the mean value of this scalar acros our stacked list)
        deviation = uv - base_line_mean
        # # now we treat the four uv scalars as a vector in 8-d space and compute the distace to the mean
        distance = np.linalg.norm(deviation, axis=(1, 3)).reshape(-1)
        # lets get the .5 cutof;
        cut_off = sorted(distance)[len(distance) // 2]
        # filter the better half
        uv_subset = uv[distance <= cut_off]
        # claculate the mean of this subset
        uv_mean = np.mean(uv_subset, axis=0)
        # use it
        self.uv_coords = uv_mean
        self.robust_uv_cords = True


def draw_frustum(img_size, K, scale=1):
    # average focal length
    f = (K[0, 0] + K[1, 1]) / 2
    # compute distances for setting up the camera pyramid
    W = 0.5 * (img_size[0])
    H = 0.5 * (img_size[1])
    Z = f
    # scale the pyramid
    W /= scale
    H /= scale
    Z /= scale
    # draw it
    glColor4f(1, 0.5, 0, 0.5)
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, 0)
    glVertex3f(-W, H, Z)
    glVertex3f(W, H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(W, H, Z)
    glVertex3f(W, -H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(W, -H, Z)
    glVertex3f(-W, -H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(-W, -H, Z)
    glVertex3f(-W, H, Z)
    glEnd()


def draw_coordinate_system(l=1):
    # Draw x-axis line.
    glColor3f(1, 0, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(l, 0, 0)
    glEnd()

    # Draw y-axis line.
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, l, 0)
    glEnd()

    # Draw z-axis line.
    glColor3f(0, 0, 1)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, l)
    glEnd()


if __name__ == "__main__":

    rotation3d = np.array([1, 2, 3], dtype=np.float32)
    translation3d = np.array([50, 60, 70], dtype=np.float32)

    # transformation from Camera Optical Center:
    #   first: translate from Camera center to object origin.
    #   second: rotate x,y,z
    #   coordinate system is x,y,z positive (not like opengl, where the z-axis is flipped.)
    # print rotation3d[0],rotation3d[1],rotation3d[2], translation3d[0],translation3d[1],translation3d[2]

    # turn translation vectors into 3x3 rot mat.
    rotation3dMat, _ = cv2.Rodrigues(rotation3d)

    # to get the transformation from object to camera we need to reverse rotation and translation
    #
    tranform3d_to_camera_translation = np.eye(4, dtype=np.float32)
    tranform3d_to_camera_translation[:-1, -1] = -translation3d

    # rotation matrix inverse == transpose
    tranform3d_to_camera_rotation = np.eye(4, dtype=np.float32)
    tranform3d_to_camera_rotation[:-1, :-1] = rotation3dMat.T

    print(tranform3d_to_camera_translation)
    print(tranform3d_to_camera_rotation)
    print(
        np.matrix(tranform3d_to_camera_rotation)
        * np.matrix(tranform3d_to_camera_translation)
    )

    # rMat, _ = cv2.Rodrigues(rotation3d)
    # self.from_camera_to_referece = np.eye(4, dtype=np.float32)
    # self.from_camera_to_referece[:-1,:-1] = rMat
    # self.from_camera_to_referece[:-1, -1] = translation3d.reshape(3)
    # # self.camera_pose_3d = np.linalg.inv(self.camera_pose_3d)
