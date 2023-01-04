"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import numpy as np
from head_pose_tracker import ui as plugin_ui
from head_pose_tracker.ui import gl_renderer_utils as utils


class HeadPoseTracker3DRenderer(plugin_ui.GLWindow):
    def __init__(
        self,
        general_settings,
        detection_storage,
        optimization_storage,
        localization_storage,
        camera_intrinsics,
        plugin,
    ):
        super().__init__(general_settings, plugin)

        self._general_settings = general_settings
        self._detection_storage = detection_storage
        self._optimization_storage = optimization_storage
        self._localization_storage = localization_storage
        self._camera_intrinsics = camera_intrinsics

    def _render(self):
        if not self._optimization_storage.calculated:
            return

        self._render_origin()
        self._render_markers()
        self._render_camera()

    def _render_origin(self):
        rotate_center_matrix = self._get_rotate_center_matrix()

        utils.render_centroid(color=(0.2, 0.2, 0.2, 0.1))
        utils.set_rotate_center(rotate_center_matrix)
        utils.render_coordinate()

    def _get_rotate_center_matrix(self):
        rotate_center_matrix = np.eye(4, dtype=np.float32)
        rotate_center_matrix[0:3, 3] = -self._optimization_storage.centroid
        return rotate_center_matrix

    def _render_markers(self):
        marker_id_to_points_3d = self._optimization_storage.marker_id_to_points_3d
        current_markers = self._detection_storage.current_markers
        current_marker_ids = [marker["id"] for marker in current_markers]

        for marker_id, points_3d in marker_id_to_points_3d.items():
            color = (
                (1, 0, 0, 0.2) if marker_id in current_marker_ids else (1, 0, 0, 0.1)
            )
            utils.render_polygon_in_3d_window(points_3d, color)

    def _render_camera(self):
        pose_data = self._localization_storage.current_pose
        camera_trace = pose_data["camera_trace"]
        camera_pose_matrix = pose_data["camera_pose_matrix"]

        # recent_camera_trace is updated no matter show_camera_trace_in_3d_window
        # is on or not
        self._localization_storage.add_recent_camera_trace(camera_trace)

        color = (0.2, 0.2, 0.2, 0.1)
        if self._general_settings.show_camera_trace_in_3d_window:
            utils.render_camera_trace(
                self._localization_storage.recent_camera_trace, color
            )

        if camera_pose_matrix is not None:
            utils.render_camera_frustum(
                camera_pose_matrix, self._camera_intrinsics, color
            )
