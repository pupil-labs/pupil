"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import csv
import logging
import os
import typing

import csv_utils
import player_methods as pm
from plugin import Plugin
from pupil_producers import Pupil_Producer_Base
from pyglui import ui
from rich.progress import track

# logging
logger = logging.getLogger(__name__)


class Raw_Data_Exporter(Plugin):
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
        should_include_low_confidence_data=True,
    ):
        super().__init__(g_pool)

        # If no pupil producer is available, don't export pupil positions
        if not self._is_pupil_producer_avaiable:
            should_export_pupil_positions = False

        self.should_export_pupil_positions = should_export_pupil_positions
        self.should_export_field_info = should_export_field_info
        self.should_export_gaze_positions = should_export_gaze_positions
        self.should_include_low_confidence_data = should_include_low_confidence_data

    def get_init_dict(self):
        return {
            "should_export_pupil_positions": self.should_export_pupil_positions,
            "should_export_field_info": self.should_export_field_info,
            "should_export_gaze_positions": self.should_export_gaze_positions,
            "should_include_low_confidence_data": self.should_include_low_confidence_data,
        }

    @property
    def _is_pupil_producer_avaiable(self) -> bool:
        producers = Pupil_Producer_Base.available_pupil_producer_plugins(self.g_pool)
        return len(producers) > 0

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Raw Data Exporter"
        self.menu.append(ui.Info_Text("Export Raw Pupil Capture data into .csv files."))
        self.menu.append(
            ui.Info_Text(
                "Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins."
            )
        )

        pupil_positions_switch = ui.Switch(
            "should_export_pupil_positions", self, label="Export Pupil Positions"
        )
        pupil_positions_switch.read_only = not self._is_pupil_producer_avaiable
        self.menu.append(pupil_positions_switch)

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
            ui.Info_Text(
                'Pupil Core software assigns "confidence" values to its pupil '
                "detections and gaze estimations. They indicate the quality of the "
                "measurement. Disable the option below to only export data above the"
                '"Minimum data confidence" threshold. This threshold can be adjusted in the '
                "general settings menu."
            )
        )
        self.menu.append(
            ui.Switch(
                "should_include_low_confidence_data",
                self,
                label="Include low confidence data",
            )
        )
        self.menu.append(
            ui.Info_Text("Press the export button or type 'e' to start the export.")
        )

    def deinit_ui(self):
        self.remove_menu()

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export_data(notification["ts_window"], notification["export_dir"])

    def export_data(self, export_window, export_dir):
        if self.should_export_pupil_positions:
            pupil_positions_exporter = Pupil_Positions_Exporter()
            pupil_positions_exporter.csv_export_write(
                positions_bisector=self.g_pool.pupil_positions[..., ...],
                timestamps=self.g_pool.timestamps,
                export_window=export_window,
                export_dir=export_dir,
                min_confidence_threshold=(
                    0.0
                    if self.should_include_low_confidence_data
                    else self.g_pool.min_data_confidence
                ),
            )

        if self.should_export_gaze_positions:
            gaze_positions_exporter = Gaze_Positions_Exporter()
            gaze_positions_exporter.csv_export_write(
                positions_bisector=self.g_pool.gaze_positions,
                timestamps=self.g_pool.timestamps,
                export_window=export_window,
                export_dir=export_dir,
                min_confidence_threshold=(
                    0.0
                    if self.should_include_low_confidence_data
                    else self.g_pool.min_data_confidence
                ),
            )

        if self.should_export_field_info:
            field_info_name = "pupil_gaze_positions_info.txt"
            field_info_path = os.path.join(export_dir, field_info_name)
            with open(field_info_path, "w", encoding="utf-8", newline="") as info_file:
                info_file.write(self.__doc__)


class _Base_Positions_Exporter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def csv_export_filename(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        pass

    @classmethod
    @abc.abstractmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_index: int
    ) -> dict:
        pass

    def csv_export_write(
        self,
        positions_bisector,
        timestamps,
        export_window,
        export_dir,
        min_confidence_threshold=0.0,
    ):
        export_file = type(self).csv_export_filename()
        export_path = os.path.join(export_dir, export_file)

        export_section = positions_bisector.init_dict_for_window(export_window)
        export_world_idc = pm.find_closest(timestamps, export_section["data_ts"])

        with open(export_path, "w", encoding="utf-8", newline="") as csvfile:
            csv_header = type(self).csv_export_labels()
            dict_writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            dict_writer.writeheader()

            for g, idx in track(
                zip(export_section["data"], export_world_idc),
                description=f"Exporting {export_file}",
                total=len(export_world_idc),
            ):
                if g["confidence"] < min_confidence_threshold:
                    continue
                dict_row = type(self).dict_export(raw_value=g, world_index=idx)
                dict_writer.writerow(dict_row)

        logger.info(f"Created '{export_file}' file.")


class Pupil_Positions_Exporter(_Base_Positions_Exporter):
    @classmethod
    def csv_export_filename(cls) -> str:
        return "pupil_positions.csv"

    @classmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        return (
            # 2d data
            "pupil_timestamp",
            "world_index",
            "eye_id",
            "confidence",
            "norm_pos_x",
            "norm_pos_y",
            "diameter",
            "method",
            # ellipse data
            "ellipse_center_x",
            "ellipse_center_y",
            "ellipse_axis_a",
            "ellipse_axis_b",
            "ellipse_angle",
            # 3d data
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

    @classmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_index: int
    ) -> dict:
        # 2d data
        pupil_timestamp = str(raw_value["timestamp"])
        eye_id = raw_value["id"]
        confidence = raw_value["confidence"]
        norm_pos_x = raw_value["norm_pos"][0]
        norm_pos_y = raw_value["norm_pos"][1]
        diameter = raw_value["diameter"]
        method = raw_value["method"]

        # ellipse data
        try:
            ellipse_center = raw_value["ellipse"]["center"]
            ellipse_axis = raw_value["ellipse"]["axes"]
            ellipse_angle = raw_value["ellipse"]["angle"]
        except KeyError:
            ellipse_center = [None, None]
            ellipse_axis = [None, None]
            ellipse_angle = None

        # 3d data
        try:
            diameter_3d = raw_value["diameter_3d"]
            model_confidence = raw_value["model_confidence"]
            sphere_center = raw_value["sphere"]["center"]
            sphere_radius = raw_value["sphere"]["radius"]
            circle_3d_center = raw_value["circle_3d"]["center"]
            circle_3d_normal = raw_value["circle_3d"]["normal"]
            circle_3d_radius = raw_value["circle_3d"]["radius"]
            theta = raw_value["theta"]
            phi = raw_value["phi"]
            projected_sphere_center = raw_value["projected_sphere"]["center"]
            projected_sphere_axis = raw_value["projected_sphere"]["axes"]
            projected_sphere_angle = raw_value["projected_sphere"]["angle"]
        except KeyError:
            diameter_3d = None
            model_confidence = None
            sphere_center = [None, None, None]
            sphere_radius = None
            circle_3d_center = [None, None, None]
            circle_3d_normal = [None, None, None]
            circle_3d_radius = None
            theta = None
            phi = None
            projected_sphere_center = [None, None]
            projected_sphere_axis = [None, None]
            projected_sphere_angle = None

        # pye3d no longer includes this field. Keeping for backwards-compatibility.
        model_id = raw_value.get("model_id", None)

        return {
            # 2d data
            "pupil_timestamp": pupil_timestamp,
            "world_index": world_index,
            "eye_id": eye_id,
            "confidence": confidence,
            "norm_pos_x": norm_pos_x,
            "norm_pos_y": norm_pos_y,
            "diameter": diameter,
            "method": method,
            # ellipse data
            "ellipse_center_x": ellipse_center[0],
            "ellipse_center_y": ellipse_center[1],
            "ellipse_axis_a": ellipse_axis[0],
            "ellipse_axis_b": ellipse_axis[1],
            "ellipse_angle": ellipse_angle,
            # 3d data
            "diameter_3d": diameter_3d,
            "model_confidence": model_confidence,
            "model_id": model_id,
            "sphere_center_x": sphere_center[0],
            "sphere_center_y": sphere_center[1],
            "sphere_center_z": sphere_center[2],
            "sphere_radius": sphere_radius,
            "circle_3d_center_x": circle_3d_center[0],
            "circle_3d_center_y": circle_3d_center[1],
            "circle_3d_center_z": circle_3d_center[2],
            "circle_3d_normal_x": circle_3d_normal[0],
            "circle_3d_normal_y": circle_3d_normal[1],
            "circle_3d_normal_z": circle_3d_normal[2],
            "circle_3d_radius": circle_3d_radius,
            "theta": theta,
            "phi": phi,
            "projected_sphere_center_x": projected_sphere_center[0],
            "projected_sphere_center_y": projected_sphere_center[1],
            "projected_sphere_axis_a": projected_sphere_axis[0],
            "projected_sphere_axis_b": projected_sphere_axis[1],
            "projected_sphere_angle": projected_sphere_angle,
        }


class Gaze_Positions_Exporter(_Base_Positions_Exporter):
    @classmethod
    def csv_export_filename(cls) -> str:
        return "gaze_positions.csv"

    @classmethod
    def csv_export_labels(cls) -> typing.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
        return (
            "gaze_timestamp",
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

    @classmethod
    def dict_export(
        cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_index: int
    ) -> dict:

        gaze_timestamp = str(raw_value["timestamp"])
        confidence = raw_value["confidence"]
        norm_pos = raw_value["norm_pos"]
        base_data = None
        gaze_points_3d = [None, None, None]
        eye_centers0_3d = [None, None, None]
        eye_centers1_3d = [None, None, None]
        gaze_normals0_3d = [None, None, None]
        gaze_normals1_3d = [None, None, None]

        if raw_value.get("base_data", None) is not None:
            base_data = raw_value["base_data"]
            base_data = " ".join(
                "{}-{}".format(b["timestamp"], b["id"]) for b in base_data
            )

        # add 3d data if avaiblable
        if raw_value.get("gaze_point_3d", None) is not None:
            gaze_points_3d = raw_value["gaze_point_3d"]
            # binocular
            if raw_value.get("eye_centers_3d", None) is not None:
                eye_centers_3d = raw_value["eye_centers_3d"]
                gaze_normals_3d = raw_value["gaze_normals_3d"]

                eye_centers0_3d = (
                    eye_centers_3d.get("0", None)
                    or eye_centers_3d.get(0, None)  # backwards compatibility
                    or [None, None, None]
                )
                eye_centers1_3d = (
                    eye_centers_3d.get("1", None)
                    or eye_centers_3d.get(1, None)  # backwards compatibility
                    or [None, None, None]
                )
                gaze_normals0_3d = (
                    gaze_normals_3d.get("0", None)
                    or gaze_normals_3d.get(0, None)  # backwards compatibility
                    or [None, None, None]
                )
                gaze_normals1_3d = (
                    gaze_normals_3d.get("1", None)
                    or gaze_normals_3d.get(1, None)  # backwards compatibility
                    or [None, None, None]
                )
            # monocular
            elif raw_value.get("eye_center_3d", None) is not None:
                try:
                    eye_id = raw_value["base_data"][0]["id"]
                except (KeyError, IndexError):
                    logger.warning(
                        f"Unexpected raw base_data for monocular gaze!"
                        f" Data: {raw_value.get('base_data', None)}"
                    )
                else:
                    if str(eye_id) == "0":
                        eye_centers0_3d = raw_value["eye_center_3d"]
                        gaze_normals0_3d = raw_value["gaze_normal_3d"]
                    elif str(eye_id) == "1":
                        eye_centers1_3d = raw_value["eye_center_3d"]
                        gaze_normals1_3d = raw_value["gaze_normal_3d"]

        return {
            "gaze_timestamp": gaze_timestamp,
            "world_index": world_index,
            "confidence": confidence,
            "norm_pos_x": norm_pos[0],
            "norm_pos_y": norm_pos[1],
            "base_data": base_data,
            "gaze_point_3d_x": gaze_points_3d[0],
            "gaze_point_3d_y": gaze_points_3d[1],
            "gaze_point_3d_z": gaze_points_3d[2],
            "eye_center0_3d_x": eye_centers0_3d[0],
            "eye_center0_3d_y": eye_centers0_3d[1],
            "eye_center0_3d_z": eye_centers0_3d[2],
            "gaze_normal0_x": gaze_normals0_3d[0],
            "gaze_normal0_y": gaze_normals0_3d[1],
            "gaze_normal0_z": gaze_normals0_3d[2],
            "eye_center1_3d_x": eye_centers1_3d[0],
            "eye_center1_3d_y": eye_centers1_3d[1],
            "eye_center1_3d_z": eye_centers1_3d[2],
            "gaze_normal1_x": gaze_normals1_3d[0],
            "gaze_normal1_y": gaze_normals1_3d[1],
            "gaze_normal1_z": gaze_normals1_3d[2],
        }
