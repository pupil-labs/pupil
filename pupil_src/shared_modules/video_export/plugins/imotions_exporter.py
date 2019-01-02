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

import csv_utils
import player_methods as pm
from methods import denormalize
from video_export.plugin_base.isolated_frame_exporter import IsolatedFrameExporter

logger = logging.getLogger(__name__)


class iMotions_Exporter(IsolatedFrameExporter):
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
        super().__init__(g_pool, max_concurrent_tasks=1)
        self.logger = logger
        self.logger.info("iMotions Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "iMotions Exporter"
        super().customize_menu()

    def export_data(self, export_range, export_dir):
        rec_start = _get_recording_start_date(self.g_pool.rec_dir)
        im_dir = os.path.join(export_dir, "iMotions_{}".format(rec_start))

        try:
            self.add_export_job(
                export_range,
                im_dir,
                input_name="world",
                output_name="scene",
                process_frame=_process_frame,
                timestamp_export_format=None,
            )
        except FileNotFoundError:
            logger.info("'world' video not found. Export continues with gaze data.")

        _copy_info_csv(self.g_pool.rec_dir, im_dir)
        _write_gaze_data(
            self.g_pool.gaze_positions,
            im_dir,
            export_range,
            self.g_pool.timestamps,
            self.g_pool.capture,
        )


def _process_frame(capture, frame):
    """
    Processing function for IsolatedFrameExporter.
    Removes camera lens distortions.
    """
    return capture.intrinsics.undistort(frame.img)


def _copy_info_csv(source_folder, destination_folder):
    info_src = os.path.join(source_folder, "info.csv")
    info_dest = os.path.join(destination_folder, "iMotions_info.csv")
    copy2(info_src, info_dest)


def _get_recording_start_date(source_folder):
    csv_loc = os.path.join(source_folder, "info.csv")
    with open(csv_loc, "r", encoding="utf-8") as csv_file:
        rec_info = csv_utils.read_key_value_file(csv_file)
        date = rec_info["Start Date"].replace(".", "_").replace(":", "_")
        time = rec_info["Start Time"].replace(":", "_")
    return "{}_{}".format(date, time)


user_warned_3d_only = False


def _write_gaze_data(
    gaze_positions, destination_folder, export_range, timestamps, capture
):
    global user_warned_3d_only
    with open(
        os.path.join(destination_folder, "gaze.tlv"), "w", encoding="utf-8", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="\t")

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
            media_timestamp = timestamps[media_idx]
            media_window = pm.enclosing_window(timestamps, media_idx)
            for gaze_pos in gaze_positions.by_ts_window(media_window):
                try:
                    pupil_dia = {}
                    for p in gaze_pos["base_data"]:
                        pupil_dia[p["id"]] = p["diameter_3d"]

                    pixel_pos = denormalize(
                        gaze_pos["norm_pos"], capture.frame_size, flip_y=True
                    )
                    undistorted3d = capture.intrinsics.unprojectPoints(pixel_pos)
                    undistorted2d = capture.intrinsics.projectPoints(
                        undistorted3d, use_distortion=False
                    )

                    data = (
                        gaze_pos["timestamp"],
                        media_timestamp,
                        media_idx - export_range[0],
                        *gaze_pos["gaze_point_3d"],  # Gaze3dX/Y/Z
                        *undistorted2d.flat,  # Gaze2dX/Y
                        pupil_dia.get(1, 0.0),  # PupilDiaLeft
                        pupil_dia.get(0, 0.0),  # PupilDiaRight
                        gaze_pos["confidence"],
                    )  # Confidence
                except KeyError:
                    if not user_warned_3d_only:
                        logger.error(
                            "Currently, the iMotions export only supports 3d gaze data"
                        )
                        user_warned_3d_only = True
                    continue
                csv_writer.writerow(data)
