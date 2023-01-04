"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import csv
import datetime
import logging
import os
from types import SimpleNamespace

import csv_utils
import player_methods as pm
from methods import denormalize
from pupil_recording import PupilRecording
from video_capture.file_backend import File_Source
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

        pupil_recording = PupilRecording(rec_dir=self.g_pool.rec_dir)
        meta = pupil_recording.meta_info

        if meta.recording_software_name == meta.RECORDING_SOFTWARE_NAME_PUPIL_INVISIBLE:
            logger.error(
                "The iMotions exporter does not yet support Pupil Invisible recordings!"
            )
            return

        rec_start = _get_recording_start_date(self.g_pool.rec_dir)
        im_dir = os.path.join(export_dir, f"iMotions_{rec_start}")

        try:
            csv_header, csv_rows = _csv_exported_gaze_data(
                gaze_positions=self.g_pool.gaze_positions,
                destination_folder=im_dir,
                export_range=export_range,
                timestamps=self.g_pool.timestamps,
                capture=self.g_pool.capture,
            )
        except _iMotionsExporterNo3DGazeDataError:
            logger.error("Currently, the iMotions export only supports 3d gaze data")
            return

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

        gaze_file_path = os.path.join(im_dir, "gaze.tlv")

        with open(gaze_file_path, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter="\t")
            csv_writer.writerow(csv_header)
            for csv_row in csv_rows:
                csv_writer.writerow(csv_row)


def _process_frame(capture, frame):
    """
    Processing function for IsolatedFrameExporter.
    Removes camera lens distortions.
    """
    return capture.intrinsics.undistort(frame.img)


def _copy_info_csv(source_folder, destination_folder):
    # TODO: The iMotions export still relies on the old-style info.csv, so we have to
    # generate this here manually. We should clarify with iMotions whether we can update
    # this to our new recording format.
    recording = PupilRecording(source_folder)
    meta = recording.meta_info

    # NOTE: This is potentially incorrect, since we don't know the timezone. But we are
    # keeping this format for backwards compatibility with the old-style info.csv.
    start_datetime = datetime.datetime.fromtimestamp(meta.start_time_system_s)
    start_date = start_datetime.strftime("%d.%m.%Y")
    start_time = start_datetime.strftime("%H:%M:%S")

    duration_full_s = meta.duration_s
    duration_h = int(duration_full_s // 3600)
    duration_m = int((duration_full_s % 3600) // 60)
    duration_s = int(round(duration_full_s % 3600 % 60))
    duration_time = f"{duration_h:02}:{duration_m:02}:{duration_s:02}"

    try:
        world_video = recording.files().core().world().videos()[0]
    except IndexError:
        logger.error("Error while exporting iMotions data. World video not found!")
        return

    cap = File_Source(SimpleNamespace(), world_video)
    world_frames = cap.get_frame_count()
    world_resolution = f"{cap.frame_size[0]}x{cap.frame_size[1]}"

    data = {}
    data["Recording Name"] = meta.recording_name
    data["Start Date"] = start_date
    data["Start Time"] = start_time
    data["Start Time (System)"] = meta.start_time_system_s
    data["Start Time (Synced)"] = meta.start_time_synced_s
    data["Recording UUID"] = str(meta.recording_uuid)
    data["Duration Time"] = duration_time
    data["World Camera Frames"] = world_frames
    data["World Camera Resolution"] = world_resolution
    data["Capture Software Version"] = meta.recording_software_version
    data["Data Format Version"] = str(meta.min_player_version)
    data["System Info"] = meta.system_info

    info_dest = os.path.join(destination_folder, "iMotions_info.csv")
    with open(info_dest, "w", newline="", encoding="utf-8") as f:
        csv_utils.write_key_value_file(f, data)


def _get_recording_start_date(source_folder):
    recording = PupilRecording(source_folder)
    meta = recording.meta_info
    # NOTE: This is potentially incorrect, since we don't know the timezone. But we are
    # keeping this format for backwards compatibility with the old-style info.csv.
    start_datetime = datetime.datetime.fromtimestamp(meta.start_time_system_s)
    return start_datetime.strftime("%d_%m_%Y_%H_%M_%S")


class _iMotionsExporterNo3DGazeDataError(Exception):
    pass


def _csv_exported_gaze_data(
    gaze_positions, destination_folder, export_range, timestamps, capture
):

    export_start, export_stop = export_range  # export_stop is exclusive
    export_window = pm.exact_window(timestamps, (export_start, export_stop - 1))
    gaze_section = gaze_positions.init_dict_for_window(export_window)

    # find closest world idx for each gaze datum
    gaze_world_idc = pm.find_closest(timestamps, gaze_section["data_ts"])

    csv_header = (
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

    csv_rows = []

    for gaze_pos, media_idx in zip(gaze_section["data"], gaze_world_idc):
        media_timestamp = timestamps[media_idx]
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
                gaze_pos["confidence"],  # Confidence
            )
        except KeyError:
            raise _iMotionsExporterNo3DGazeDataError()

        csv_rows.append(data)

    return csv_header, csv_rows
