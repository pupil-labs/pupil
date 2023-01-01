"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import numpy as np
from OpenGL import GL as gl
from pyglui.cygl import utils as cygl_utils
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path


class DetectionRenderer:
    """
    Renders 2d marker locations in the world video.
    """

    def __init__(
        self, general_settings, detection_storage, optimization_storage, plugin
    ):
        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage

        self._square_definition = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
        )
        self._hat_definition = np.array(
            [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
        )

        self._setup_glfont()

        plugin.add_observer("gl_display", self._on_gl_display)

    def _setup_glfont(self):
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", get_opensans_font_path())
        self.glfont.set_size(20)
        self.glfont.set_color_float((0.8, 0.2, 0.1, 0.8))

    def _on_gl_display(self):
        self._render()

    def _render(self):
        if not self._general_settings.render_markers:
            return
        current_markers = self._detection_storage.current_markers
        marker_id_optimized = self._get_marker_id_optimized()
        self._render_markers(current_markers, marker_id_optimized)

    def _get_marker_id_optimized(self):
        try:
            return self._optimization_storage.marker_id_to_extrinsics.keys()
        except TypeError:
            return []

    def _render_markers(self, current_markers, marker_id_optimized):
        for marker in current_markers:
            marker_points = np.array(marker["verts"], dtype=np.float32)
            hat_points = self._calculate_hat_points(marker_points)
            if marker["id"] in marker_id_optimized:
                color = (1.0, 0.0, 0.0, 0.2)
            else:
                color = (0.0, 1.0, 1.0, 0.2)

            self._draw_hat(hat_points, color)

            if self._general_settings.show_marker_id_in_main_window:
                self._draw_marker_id(marker_points, marker["id"])

    def _calculate_hat_points(self, marker_points):
        perspective_matrix = cv2.getPerspectiveTransform(
            self._square_definition, marker_points
        )
        hat_points = cv2.perspectiveTransform(self._hat_definition, perspective_matrix)
        hat_points.shape = 6, 2
        return hat_points

    @staticmethod
    def _draw_hat(points, color):
        cygl_utils.draw_polyline(points, 1, cygl_utils.RGBA(*color), gl.GL_POLYGON)

    def _draw_marker_id(self, marker_points, marker_id):
        point = np.max(marker_points, axis=0)
        self.glfont.draw_text(point[0], point[1], str(marker_id))
