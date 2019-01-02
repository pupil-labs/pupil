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

from pyglui import ui

import player_methods as pm
from plugin import Analysis_Plugin_Base

# logging
logger = logging.getLogger(__name__)


class Raw_Data_Exporter(Analysis_Plugin_Base):
    """
    pupil_positions.csv
    keys:
        timestamp - timestamp of the source image frame
        index - associated_frame: closest world video frame
        id - 0 or 1 for right and left eye (from the wearer's point of view)
        confidence - is an assessment by the pupil detector on how sure we can be on this measurement. A value of `0` indicates no confidence. `1` indicates perfect confidence. In our experience usefull data carries a confidence value greater than ~0.6. A `confidence` of exactly `0` means that we don't know anything. So you should ignore the position data.        norm_pos_x - x position in the eye image frame in normalized coordinates
        norm_pos_x - x position in the eye image frame in normalized coordinates
        norm_pos_y - y position in the eye image frame in normalized coordinates
        diameter - diameter of the pupil in image pixels as observed in the eye image frame (is not corrected for perspective)

        method - string that indicates what detector was used to detect the pupil

        --- optional fields depending on detector

        #in 2d the pupil appears as an ellipse available in `3d c++` and `2D c++` detector
        ellipse_center_x - x center of the pupil in image pixels
        ellipse_center_y - y center of the pupil in image pixels
        ellipse_axis_a - first axis of the pupil ellipse in pixels
        ellipse_axis_b - second axis of the pupil ellipse in pixels
        ellipse_angle - angle of the ellipse in degrees


        #data made available by the `3d c++` detector

        diameter_3d - diameter of the pupil scaled to mm based on anthropomorphic avg eye ball diameter and corrected for perspective.
        model_confidence - confidence of the current eye model (0-1)
        model_id - id of the current eye model. When a slippage is detected the model is replaced and the id changes.

        sphere_center_x - x pos of the eyeball sphere is eye pinhole camera 3d space units are scaled to mm.
        sphere_center_y - y pos of the eye ball sphere
        sphere_center_z - z pos of the eye ball sphere
        sphere_radius - radius of the eyeball. This is always 12mm (the anthropomorphic avg.) We need to make this assumption because of the `single camera scale ambiguity`.

        circle_3d_center_x - x center of the pupil as 3d circle in eye pinhole camera 3d space units are mm.
        circle_3d_center_y - y center of the pupil as 3d circle
        circle_3d_center_z - z center of the pupil as 3d circle
        circle_3d_normal_x - x normal of the pupil as 3d circle. Indicates the direction that the pupil points at in 3d space.
        circle_3d_normal_y - y normal of the pupil as 3d circle
        circle_3d_normal_z - z normal of the pupil as 3d circle
        circle_3d_radius - radius of the pupil as 3d circle. Same as `diameter_3d`

        theta - circle_3d_normal described in spherical coordinates
        phi - circle_3d_normal described in spherical coordinates

        projected_sphere_center_x - x center of the 3d sphere projected back onto the eye image frame. Units are in image pixels.
        projected_sphere_center_y - y center of the 3d sphere projected back onto the eye image frame
        projected_sphere_axis_a - first axis of the 3d sphere projection.
        projected_sphere_axis_b - second axis of the 3d sphere projection.
        projected_sphere_angle - angle of the 3d sphere projection. Units are degrees.


    gaze_positions.csv
    keys:
        timestamp - timestamp of the source image frame
        index - associated_frame: closest world video frame
        confidence - computed confidence between 0 (not confident) -1 (confident)
        norm_pos_x - x position in the world image frame in normalized coordinates
        norm_pos_y - y position in the world image frame in normalized coordinates
        base_data - "timestamp-id timestamp-id ..." of pupil data that this gaze position is computed from

        #data made available by the 3d vector gaze mappers
        gaze_point_3d_x - x position of the 3d gaze point (the point the sublejct lookes at) in the world camera coordinate system
        gaze_point_3d_y - y position of the 3d gaze point
        gaze_point_3d_z - z position of the 3d gaze point
        eye_center0_3d_x - x center of eye-ball 0 in the world camera coordinate system (of camera 0 for binocular systems or any eye camera for monocular system)
        eye_center0_3d_y - y center of eye-ball 0
        eye_center0_3d_z - z center of eye-ball 0
        gaze_normal0_x - x normal of the visual axis for eye 0 in the world camera coordinate system (of eye 0 for binocular systems or any eye for monocular system). The visual axis goes through the eye ball center and the object thats looked at.
        gaze_normal0_y - y normal of the visual axis for eye 0
        gaze_normal0_z - z normal of the visual axis for eye 0
        eye_center1_3d_x - x center of eye-ball 1 in the world camera coordinate system (not avaible for monocular setups.)
        eye_center1_3d_y - y center of eye-ball 1
        eye_center1_3d_z - z center of eye-ball 1
        gaze_normal1_x - x normal of the visual axis for eye 1 in the world camera coordinate system (not avaible for monocular setups.). The visual axis goes through the eye ball center and the object thats looked at.
        gaze_normal1_y - y normal of the visual axis for eye 1
        gaze_normal1_z - z normal of the visual axis for eye 1
        """

    icon_chr = chr(0xE873)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        should_export_pupil_positions=True,
        should_export_field_info=True,
        should_export_gaze_positions=True,
    ):
        super().__init__(g_pool)
        self.should_export_pupil_positions = should_export_pupil_positions
        self.should_export_field_info = should_export_field_info
        self.should_export_gaze_positions = should_export_gaze_positions

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Raw Data Exporter"
        self.menu.append(ui.Info_Text("Export Raw Pupil Capture data into .csv files."))
        self.menu.append(
            ui.Info_Text(
                "Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins."
            )
        )
        self.menu.append(
            ui.Switch(
                "should_export_pupil_positions", self, label="Export Pupil Positions"
            )
        )
        self.menu.append(
            ui.Switch(
                "should_export_field_info",
                self,
                label="Export Pupil Gaze Positions Info",
            )
        )
        self.menu.append(
            ui.Switch(
                "should_export_gaze_positions", self, label="Export Gaze Positions"
            )
        )
        self.menu.append(
            ui.Info_Text("Press the export button or type 'e' to start the export.")
        )

    def deinit_ui(self):
        self.remove_menu()

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export_data(notification["range"], notification["export_dir"])

    def export_data(self, export_range, export_dir):
        export_window = pm.exact_window(self.g_pool.timestamps, export_range)
        if self.should_export_pupil_positions:
            with open(
                os.path.join(export_dir, "pupil_positions.csv"),
                "w",
                encoding="utf-8",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=",")

                csv_writer.writerow(
                    (
                        "world_timestamp",
                        "world_index",
                        "eye_id",
                        "confidence",
                        "norm_pos_x",
                        "norm_pos_y",
                        "diameter",
                        "method",
                        "ellipse_center_x",
                        "ellipse_center_y",
                        "ellipse_axis_a",
                        "ellipse_axis_b",
                        "ellipse_angle",
                        "diameter_3d",
                        "model_confidence",
                        "model_id",
                        "sphere_center_x",
                        "sphere_center_y",
                        "sphere_center_z",
                        "sphere_radius",
                        "circle_3d_center_x",
                        "circle_3d_center_y",
                        "circle_3d_center_z",
                        "circle_3d_normal_x",
                        "circle_3d_normal_y",
                        "circle_3d_normal_z",
                        "circle_3d_radius",
                        "theta",
                        "phi",
                        "projected_sphere_center_x",
                        "projected_sphere_center_y",
                        "projected_sphere_axis_a",
                        "projected_sphere_axis_b",
                        "projected_sphere_angle",
                    )
                )

                pupil_section = self.g_pool.pupil_positions.init_dict_for_window(
                    export_window
                )
                pupil_world_idc = pm.find_closest(
                    self.g_pool.timestamps, pupil_section["data_ts"]
                )
                for p, idx in zip(pupil_section["data"], pupil_world_idc):
                    data_2d = [
                        "{}".format(
                            p["timestamp"]
                        ),  # use str to be consitant with csv lib.
                        idx,
                        p["id"],
                        p["confidence"],
                        p["norm_pos"][0],
                        p["norm_pos"][1],
                        p["diameter"],
                        p["method"],
                    ]
                    try:
                        ellipse_data = [
                            p["ellipse"]["center"][0],
                            p["ellipse"]["center"][1],
                            p["ellipse"]["axes"][0],
                            p["ellipse"]["axes"][1],
                            p["ellipse"]["angle"],
                        ]
                    except KeyError:
                        ellipse_data = [None] * 5
                    try:
                        data_3d = [
                            p["diameter_3d"],
                            p["model_confidence"],
                            p["model_id"],
                            p["sphere"]["center"][0],
                            p["sphere"]["center"][1],
                            p["sphere"]["center"][2],
                            p["sphere"]["radius"],
                            p["circle_3d"]["center"][0],
                            p["circle_3d"]["center"][1],
                            p["circle_3d"]["center"][2],
                            p["circle_3d"]["normal"][0],
                            p["circle_3d"]["normal"][1],
                            p["circle_3d"]["normal"][2],
                            p["circle_3d"]["radius"],
                            p["theta"],
                            p["phi"],
                            p["projected_sphere"]["center"][0],
                            p["projected_sphere"]["center"][1],
                            p["projected_sphere"]["axes"][0],
                            p["projected_sphere"]["axes"][1],
                            p["projected_sphere"]["angle"],
                        ]
                    except KeyError:
                        data_3d = [None] * 21
                    row = data_2d + ellipse_data + data_3d
                    csv_writer.writerow(row)
                logger.info("Created 'pupil_positions.csv' file.")

        if self.should_export_gaze_positions:
            with open(
                os.path.join(export_dir, "gaze_positions.csv"),
                "w",
                encoding="utf-8",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=",")
                csv_writer.writerow(
                    (
                        "world_timestamp",
                        "world_index",
                        "confidence",
                        "norm_pos_x",
                        "norm_pos_y",
                        "base_data",
                        "gaze_point_3d_x",
                        "gaze_point_3d_y",
                        "gaze_point_3d_z",
                        "eye_center0_3d_x",
                        "eye_center0_3d_y",
                        "eye_center0_3d_z",
                        "gaze_normal0_x",
                        "gaze_normal0_y",
                        "gaze_normal0_z",
                        "eye_center1_3d_x",
                        "eye_center1_3d_y",
                        "eye_center1_3d_z",
                        "gaze_normal1_x",
                        "gaze_normal1_y",
                        "gaze_normal1_z",
                    )
                )

                gaze_section = self.g_pool.gaze_positions.init_dict_for_window(
                    export_window
                )
                gaze_world_idc = pm.find_closest(
                    self.g_pool.timestamps, gaze_section["data_ts"]
                )

                for g, idx in zip(gaze_section["data"], gaze_world_idc):
                    data = [
                        "{}".format(g["timestamp"]),
                        idx,
                        g["confidence"],
                        g["norm_pos"][0],
                        g["norm_pos"][1],
                        " ".join(
                            [
                                "{}-{}".format(b["timestamp"], b["id"])
                                for b in g["base_data"]
                            ]
                        ),
                    ]  # use str on timestamp to be consitant with csv lib.

                    # add 3d data if avaiblable
                    if g.get("gaze_point_3d", None) is not None:
                        data_3d = [
                            g["gaze_point_3d"][0],
                            g["gaze_point_3d"][1],
                            g["gaze_point_3d"][2],
                        ]

                        # binocular
                        if g.get("eye_centers_3d", None) is not None:
                            data_3d += g["eye_centers_3d"].get(0, [None, None, None])
                            data_3d += g["gaze_normals_3d"].get(0, [None, None, None])
                            data_3d += g["eye_centers_3d"].get(1, [None, None, None])
                            data_3d += g["gaze_normals_3d"].get(1, [None, None, None])
                        # monocular
                        elif g.get("eye_center_3d", None) is not None:
                            data_3d += g["eye_center_3d"]
                            data_3d += g["gaze_normal_3d"]
                            data_3d += [None] * 6
                    else:
                        data_3d = [None] * 15
                    data += data_3d
                    csv_writer.writerow(data)
                logger.info("Created 'gaze_positions.csv' file.")
        if self.should_export_field_info:
            with open(
                os.path.join(export_dir, "pupil_gaze_positions_info.txt"),
                "w",
                encoding="utf-8",
                newline="",
            ) as info_file:
                info_file.write(self.__doc__)
