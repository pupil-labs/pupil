"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from pupil_detector_plugins import color_scheme
from pyglui import ui
from video_export.plugin_base.isolated_frame_exporter import IsolatedFrameExporter
from video_overlay.utils.image_manipulation import PupilRenderer

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
        self.menu.append(
            ui.Switch("render_pupil", self, label="Visualize Pupil Detection")
        )
        self.menu.append(ui.Info_Text("Color Legend"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_2D.as_float, "2D pupil ellipse")
        )
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_3D.as_float, "3D pupil ellipse")
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_IN.as_float,
                "Long-term model outline (within bounds)",
            )
        )
        self.menu.append(
            ui.Color_Legend(
                color_scheme.EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_OUT.as_float,
                "Long-term model outline (out-of-bounds)",
            )
        )
        self.menu.append(ui.Separator())
        super().customize_menu()

    def _export_eye_video(self, export_range, export_dir, eye_id):
        if self.render_pupil:
            process_frame = _add_pupil_ellipse(
                self.g_pool.pupil_positions[eye_id, "2d"],
                self.g_pool.pupil_positions[eye_id, "3d"],
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

    _2D = "2d"
    _3D = "3d"

    def __init__(self, pupil_positions_of_eye_2d, pupil_positions_of_eye_3d):
        self._warned_once_data_not_found = {self._2D: False, self._3D: False}

        self.renderer = PupilRenderer(pupil_getter=None)
        self._render_functions = {
            self._2D: self.renderer.render_pupil_2d,
            self._3D: self.renderer.render_pupil_3d,
        }

        self._pupil_positions_of_eye = {
            self._2D: pupil_positions_of_eye_2d,
            self._3D: pupil_positions_of_eye_3d,
        }

    def __call__(self, _, frame):
        eye_image = frame.img
        timestamp = frame.timestamp
        self._render(eye_image, timestamp, self._2D)
        self._render(eye_image, timestamp, self._3D)
        return eye_image

    def _render(self, image, timestamp, mode):
        try:
            pupil_datum = self._pupil_positions_of_eye[mode].by_ts(timestamp)
            self._render_functions[mode](image, pupil_datum)
        except ValueError:
            if not self._warned_once_data_not_found[mode]:
                logger.warning(f"No {mode} data for pupil visualization found.")
                self._warned_once_data_not_found[mode] = True
