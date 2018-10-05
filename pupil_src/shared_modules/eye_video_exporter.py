"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

from pyglui import ui

from video_exporter import VideoExporter
from vis_eye_video_overlay import draw_pupil_on_image

logger = logging.getLogger(__name__)


def _no_change(_, frame):
    return frame.img


def _add_pupil_ellipse(pupil_positions_of_eye):
    def add_pupil_ellipse(_, frame):
        eye_image = frame.img
        try:
            pupil_datum = pupil_positions_of_eye.by_ts(frame.timestamp)
            draw_pupil_on_image(eye_image, pupil_datum)
        except ValueError:
            logger.warning("Inconsistent timestamps found in pupil data")
        return eye_image

    return add_pupil_ellipse


class Eye_Video_Exporter(VideoExporter):
    """Eye Video Exporter

    All files exported by this plugin are saved to a subdirectory within
    the export directory called "EyeVideo".
    """

    icon_chr = "EV"

    def __init__(self, g_pool, render_pupil=True):
        super().__init__(g_pool)
        self.render_pupil = render_pupil
        logger.info("Eye Video Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "Eye Video Exporter"
        self.menu.append(ui.Switch("render_pupil", self, label="Render detected pupil"))

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
                plugin_name="EyeVideo",
                input_name=eye_name,
                output_name=eye_name,
                process_frame=process_frame,
            )
        except FileNotFoundError:
            # happens if there is no such eye video
            pass

    def export_data(self, export_range, export_dir):
        self._export_eye_video(export_range, export_dir, eye_id=0)
        self._export_eye_video(export_range, export_dir, eye_id=1)
