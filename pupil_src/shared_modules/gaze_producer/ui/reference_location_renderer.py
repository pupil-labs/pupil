"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import gl_utils
import pyglui.cygl.utils as cygl_utils


class ReferenceLocationRenderer:
    """
    Renders reference locations in the world video.
    """

    # we not only draw reference locations in the current frame, but also from close
    # frames in a certain range
    close_ref_range = 4

    def __init__(
        self, reference_location_storage, plugin, frame_size, get_current_frame_index
    ):
        self._reference_location_storage = reference_location_storage

        self._frame_size = frame_size
        self._get_current_frame_index = get_current_frame_index

        plugin.add_observer("gl_display", self.on_gl_display)

    def on_gl_display(self):
        self._render_reference_locations()

    def _render_reference_locations(self):
        current_index = self._get_current_frame_index()
        current_reference = self._reference_location_storage.get_or_none(current_index)
        if current_reference:
            self._draw_current_reference(current_reference)
        else:
            # first reference before
            self._draw_first_reference_in_range(
                range(current_index - 1, current_index - self.close_ref_range - 1, -1),
                current_index,
            )
            # first reference after
            self._draw_first_reference_in_range(
                range(current_index + 1, current_index + self.close_ref_range + 1),
                current_index,
            )

    def _draw_current_reference(self, current_reference):
        with self._frame_coordinate_system:
            cygl_utils.draw_points(
                [current_reference.screen_pos],
                size=35,
                color=cygl_utils.RGBA(0, 0.5, 0.5, 0.7),
            )
            self._draw_inner_dot(current_reference)

    def _draw_first_reference_in_range(self, range_, current_index):
        for index in range_:
            reference_location = self._reference_location_storage.get_or_none(index)
            if reference_location:
                diff_to_current = abs(current_index - index)
                self._draw_close_reference(reference_location, diff_to_current)
                break

    def _draw_close_reference(self, reference_location, diff_to_current):
        with self._frame_coordinate_system:
            alpha = 0.7 * (1.0 - diff_to_current / (self.close_ref_range + 1.0))
            cygl_utils.draw_progress(
                reference_location.screen_pos,
                0.0,
                0.999,
                inner_radius=20.0,
                outer_radius=35.0,
                color=cygl_utils.RGBA(0, 0.5, 0.5, alpha),
            )
            self._draw_inner_dot(reference_location)

    @property
    def _frame_coordinate_system(self):
        return gl_utils.Coord_System(
            left=0, right=self._frame_size[0], bottom=self._frame_size[1], top=0
        )

    @staticmethod
    def _draw_inner_dot(reference_location):
        cygl_utils.draw_points(
            [reference_location.screen_pos],
            size=5,
            color=cygl_utils.RGBA(0.0, 0.9, 0.0, 1.0),
        )
