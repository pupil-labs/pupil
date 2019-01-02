"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import csv
import logging
import os
from shutil import copy2

import player_methods as pm
from methods import denormalize
from video_exporter import VideoExporter

logger = logging.getLogger(__name__)


def _process_frame(capture, frame):
    return capture.intrinsics.undistort(frame.img)


class iMotions_Exporter(VideoExporter):
    """iMotions Gaze and Video Exporter

    All files exported by this plugin are saved to a subdirectory within
    the export directory called "iMotions". The gaze data will be written
    into a file called "gaze.tlv" and the undistorted scene video will be
    saved in a file called "scene.mp4".

    The gaze.tlv file is a tab-separated CSV file with the following fields:
        GazeTimeStamp: Timestamp of the gaze point, unit: seconds
        MediaTimeStamp: Timestamp of the scene frame to which the gaze point
                        was correlated to, unit: seconds
        MediaFrameIndex: Index of the scene frame to which the gaze point was
                         correlated to
        Gaze3dX: X position of the 3d gaze point (the point the subject looks
                 at) in the scene camera coordinate system
        Gaze3dY: Y position of the 3d gaze point
        Gaze3dZ: Z position of the 3d gaze point
        Gaze2dX: undistorted gaze pixel position, X coordinate, unit: pixels
        Gaze2dX: undistorted gaze pixel position, Y coordinate, unit: pixels
        PupilDiaLeft: Left pupil diameter, 0.0 if not available, unit: millimeters
        PupilDiaRight: Right pupil diameter, 0.0 if not available, unit: millimeters
        Confidence: Value between 0 and 1 indicating the quality of the gaze
                    datum. It depends on the confidence of the pupil detection
                    and the confidence of the 3d model. Higher values are good.
    """

    icon_chr = "iM"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        logger.info("iMotions Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "iMotions Exporter"

    def export_data(self, export_range, export_dir):
        user_warned_3d_only = False

        rec_start = self._get_recording_start_date()
        im_dir = os.path.join(export_dir, "iMotions_{}".format(rec_start))

        try:
            self.add_export_job(
                export_range,
                im_dir,
                plugin_name="iMotions",
                input_name="world",
                output_name="scene",
                process_frame=_process_frame,
                export_timestamps=False,
            )
        except FileNotFoundError:
            logger.info("'world' video not found. Export continues with gaze data.")

        info_src = os.path.join(self.g_pool.rec_dir, "info.csv")
        info_dest = os.path.join(im_dir, "iMotions_info.csv")
        copy2(info_src, info_dest)  # copy info.csv file

        with open(
            os.path.join(im_dir, "gaze.tlv"), "w", encoding="utf-8", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile, delimiter="\t")

            csv_writer.writerow(
                (
                    "GazeTimeStamp",
                    "MediaTimeStamp",
                    "MediaFrameIndex",
                    "Gaze3dX",
                    "Gaze3dY",
                    "Gaze3dZ",
                    "Gaze2dX",
                    "Gaze2dY",
                    "PupilDiaLeft",
                    "PupilDiaRight",
                    "Confidence",
                )
            )

            for media_idx in range(*export_range):
                media_timestamp = self.g_pool.timestamps[media_idx]
                media_window = pm.enclosing_window(self.g_pool.timestamps, media_idx)
                for g in self.g_pool.gaze_positions.by_ts_window(media_window):
                    try:
                        pupil_dia = {}
                        for p in g["base_data"]:
                            pupil_dia[p["id"]] = p["diameter_3d"]

                        pixel_pos = denormalize(
                            g["norm_pos"], self.g_pool.capture.frame_size, flip_y=True
                        )
                        undistorted3d = self.g_pool.capture.intrinsics.unprojectPoints(
                            pixel_pos
                        )
                        undistorted2d = self.g_pool.capture.intrinsics.projectPoints(
                            undistorted3d, use_distortion=False
                        )

                        data = (
                            g["timestamp"],
                            media_timestamp,
                            media_idx - export_range[0],
                            *g["gaze_point_3d"],  # Gaze3dX/Y/Z
                            *undistorted2d.flat,  # Gaze2dX/Y
                            pupil_dia.get(1, 0.0),  # PupilDiaLeft
                            pupil_dia.get(0, 0.0),  # PupilDiaRight
                            g["confidence"],
                        )  # Confidence
                    except KeyError:
                        if not user_warned_3d_only:
                            logger.error(
                                "Currently, the iMotions export only supports 3d gaze data"
                            )
                            user_warned_3d_only = True
                        continue
                    csv_writer.writerow(data)
