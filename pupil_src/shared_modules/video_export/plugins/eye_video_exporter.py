"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

from pyglui import ui

from video_export.plugin_base.isolated_frame_exporter import IsolatedFrameExporter
from vis_eye_video_overlay import draw_pupil_on_image


logger = logging.getLogger(__name__)


class Eye_Video_Exporter(IsolatedFrameExporter):
    """
    Exports eye videos in the selected time range together with their timestamps.
    Optionally (via a switch button), pupil detections are rendered on the video.
    """

    icon_chr = "EV"

    def __init__(self, g_pool, render_pupil=True):
        super().__init__(g_pool, max_concurrent_tasks=2)  # export 2 eyes at once
        self.render_pupil = render_pupil
        self.logger = logger
        self.logger.info("Eye Video Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "Eye Video Exporter"
        self.menu.append(ui.Switch("render_pupil", self, label="Render detected pupil"))
        super().customize_menu()

    def _export_eye_video(self, export_range, export_dir, eye_id):
        if self.render_pupil:
            process_frame = _add_pupil_ellipse(
                self.g_pool.pupil_positions_by_id[eye_id]
            )
        else:
            process_frame = _no_change
        eye_name = "eye" + str(eye_id)
        try:
            self.add_export_job(
                export_range,
                export_dir,
                input_name=eye_name,
                output_name=eye_name,
                process_frame=process_frame,
                timestamp_export_format="all",
            )
        except FileNotFoundError:
            # happens if there is no such eye video
            pass

    def export_data(self, export_range, export_dir):
        self._export_eye_video(export_range, export_dir, eye_id=0)
        self._export_eye_video(export_range, export_dir, eye_id=1)


def _no_change(_, frame):
    """
    Processing function for IsolatedFrameExporter.
    Just leaves all frames unchanged.
    """
    return frame.img


class _add_pupil_ellipse:
    """
    Acts as a processing function for IsolatedFrameExporter.
    Renders pupil detection on top of eye images

    This is a class because we need to store all
    pupil positions for rendering.
    """

    _warned_once_data_not_found = False

    def __init__(self, pupil_positions_of_eye):
        self._pupil_positions_of_eye = pupil_positions_of_eye

    def __call__(self, _, frame):
        eye_image = frame.img
        try:
            pupil_datum = self._pupil_positions_of_eye.by_ts(frame.timestamp)
            draw_pupil_on_image(eye_image, pupil_datum)
        except ValueError:
            if not self._warned_once_data_not_found:
                logger.warning("Could not draw pupil visualization. No data found.")
                self._warned_once_data_not_found = True
        return eye_image
