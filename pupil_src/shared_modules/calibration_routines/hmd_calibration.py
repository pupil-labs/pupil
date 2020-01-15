"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os

import numpy as np
from pyglui import ui

import audio
from calibration_routines import calibrate
from calibration_routines.calibration_plugin_base import Calibration_Plugin
from calibration_routines.finish_calibration import (
    SphericalCamera,
    not_enough_data_error_msg,
    solver_failed_to_converge_error_msg,
)
from calibration_routines.optimization_calibration import utils, BundleAdjustment
from file_methods import save_object

logger = logging.getLogger(__name__)


class HMD_Calibration(Calibration_Plugin):
    """Calibrate gaze on HMD screen.
    """

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.info = None
        self.menu = None

    def init_ui(self):
        self.add_menu()
        self.menu.label = "HMD Calibration"
        self.menu.append(ui.Info_Text("Calibrate gaze parameters to map onto an HMD."))
        self.calib_button = ui.Thumb(
            "active", self, label="C", setter=self.toggle_calibration, hotkey="c"
        )

    def deinit_ui(self):
        if self.active:
            self.stop()
        self.remove_menu()

    def on_notify(self, notification):
        """Calibrates user gaze for HMDs

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``:  Stops the calibration procedure
           ``calibration.add_ref_data``: Adds reference data

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped

        Args:
            notification (dictionary): Notification dictionary
        """
        try:
            if notification["subject"].startswith("calibration.should_start"):
                if self.active:
                    logger.warning("Calibration already running.")
                else:
                    hmd_video_frame_size = notification["hmd_video_frame_size"]
                    outlier_threshold = notification["outlier_threshold"]
                    self.start(hmd_video_frame_size, outlier_threshold)
            elif notification["subject"].startswith("calibration.should_stop"):
                if self.active:
                    self.stop()
                else:
                    logger.warning("Calibration already stopped.")
            elif notification["subject"].startswith("calibration.add_ref_data"):
                if self.active:
                    self.ref_list += notification["ref_data"]
                else:
                    logger.error(
                        "Ref data can only be added when calibration is running."
                    )
        except KeyError as e:
            logger.error(
                "Notification: {} not conform. Raised error {}".format(notification, e)
            )

    def start(self, hmd_video_frame_size, outlier_threshold):
        audio.say("Starting Calibration")
        logger.info("Starting Calibration")
        self.active = True
        self.pupil_list = []
        self.ref_list = []
        self.hmd_video_frame_size = hmd_video_frame_size
        self.outlier_threshold = outlier_threshold
        try:
            self.g_pool.quickbar.insert(0, self.calib_button)
        except AttributeError:
            pass  # quickbar and calib_button are not defined in Service
        self.notify_all({"subject": "calibration.started"})

    def stop(self):
        audio.say("Stopping Calibration")
        logger.info("Stopping Calibration")
        self.active = False
        self.finish_calibration()
        try:
            self.g_pool.quickbar.remove(self.calib_button)
        except AttributeError:
            pass  # quickbar and calib_button are not defined in Service
        self.notify_all({"subject": "calibration.stopped"})

    def finish_calibration(self):
        pupil_list = self.pupil_list
        ref_list = self.ref_list
        hmd_video_frame_size = self.hmd_video_frame_size

        g_pool = self.g_pool

        pupil0 = [p for p in pupil_list if p["id"] == 0]
        pupil1 = [p for p in pupil_list if p["id"] == 1]

        ref0 = [r for r in ref_list if r["id"] == 0]
        ref1 = [r for r in ref_list if r["id"] == 1]

        matched_pupil0_data = calibrate.closest_matches_monocular(ref0, pupil0)
        matched_pupil1_data = calibrate.closest_matches_monocular(ref1, pupil1)

        if matched_pupil0_data:
            cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_pupil0_data)
            map_fn0, inliers0, params0 = calibrate.calibrate_2d_polynomial(
                cal_pt_cloud, hmd_video_frame_size, binocular=False
            )
            if not inliers0.any():
                self.notify_all(
                    {
                        "subject": "calibration.failed",
                        "reason": solver_failed_to_converge_error_msg,
                    }
                )
                return
        else:
            logger.warning("No matched ref<->pupil data collected for id0")
            params0 = None

        if matched_pupil1_data:
            cal_pt_cloud = calibrate.preprocess_2d_data_monocular(matched_pupil1_data)
            map_fn1, inliers1, params1 = calibrate.calibrate_2d_polynomial(
                cal_pt_cloud, hmd_video_frame_size, binocular=False
            )
            if not inliers1.any():
                self.notify_all(
                    {
                        "subject": "calibration.failed",
                        "reason": solver_failed_to_converge_error_msg,
                    }
                )
                return
        else:
            logger.warning("No matched ref<->pupil data collected for id1")
            params1 = None

        if params0 or params1:
            ts = g_pool.get_timestamp()
            if params0 and params1:
                method = "dual monocular polynomial regression"
                mapper = "Dual_Monocular_Gaze_Mapper"
                args = {"params0": params0, "params1": params1}
            elif params0:
                method = "monocular polynomial regression"
                mapper = "Monocular_Gaze_Mapper"
                args = {"params": params0}
            elif params1:
                method = "monocular polynomial regression"
                mapper = "Monocular_Gaze_Mapper"
                args = {"params": params1}

            # Announce success
            self.notify_all(
                {
                    "subject": "calibration.successful",
                    "method": method,
                    "timestamp": ts,
                    "record": True,
                }
            )

            # Announce calibration data
            self.notify_all(
                {
                    "subject": "calibration.calibration_data",
                    "timestamp": ts,
                    "pupil_list": pupil_list,
                    "ref_list": ref_list,
                    "calibration_method": method,
                    "record": True,
                }
            )

            # Start mapper
            self.notify_all({"subject": "start_plugin", "name": mapper, "args": args})
        else:
            logger.error("Calibration failed for both eyes. No data found")
            self.notify_all(
                {"subject": "calibration.failed", "reason": not_enough_data_error_msg}
            )

    def recent_events(self, events):
        if self.active:
            self.pupil_list.extend(events["pupil"])

    def get_init_dict(self):
        d = {}
        return d


class HMD_Calibration_3D(HMD_Calibration, Calibration_Plugin):
    """docstring for HMD 3d calibratoin"""

    def __init__(self, g_pool):
        super(HMD_Calibration_3D, self).__init__(g_pool)
        self.eye_translations = [0, 0, 0], [0, 0, 0]  # overwritten on start_calibrate

    def on_notify(self, notification):
        """Calibrates user gaze for HMDs

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``:  Stops the calibration procedure
           ``calibration.add_ref_data``: Adds reference data

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped

        Args:
            notification (dictionary): Notification dictionary
        """
        try:
            if notification["subject"].startswith("calibration.should_start"):
                if self.active:
                    logger.warning("Calibration already running.")
                else:
                    assert len(notification["translation_eye0"]) == 3
                    assert len(notification["translation_eye1"]) == 3
                    self.eye_translations = (
                        notification["translation_eye0"],
                        notification["translation_eye1"],
                    )
                    self.start(None, None)
            elif notification["subject"].startswith("calibration.should_stop"):
                if self.active:
                    self.stop()
                else:
                    logger.warning("Calibration already stopped.")
            elif notification["subject"].startswith("calibration.add_ref_data"):
                if self.active:
                    self.ref_list += notification["ref_data"]
                else:
                    logger.error(
                        "Ref data can only be added when calibration is running."
                    )
        except KeyError as e:
            logger.error(
                "Notification: %s not conform. Raised error %s" % (notification, e)
            )

    def finish_calibration(self):
        pupil_list = self.pupil_list
        ref_list = self.ref_list
        g_pool = self.g_pool

        matched_data = calibrate.closest_matches_binocular(ref_list, pupil_list)
        save_object(matched_data, "hmd_cal_data")

        ref_points_3d_unscaled = [d["ref"]["mm_pos"] for d in matched_data]
        pupil0_normals = [
            d["pupil"]["circle_3d"]["normal"]
            for d in matched_data
            if "3d" in d["pupil"]["method"]
        ]
        pupil1_normals = [
            d["pupil1"]["circle_3d"]["normal"]
            for d in matched_data
            if "3d" in d["pupil"]["method"]
        ]

        not_enough_data_error_msg = "Did not collect enough data during calibration."
        if (
            len(ref_points_3d_unscaled) < 1
            or len(pupil0_normals) < 1
            or len(pupil1_normals) < 1
        ):
            logger.error(not_enough_data_error_msg)
            self.notify_all(
                {
                    "subject": "calibration.failed",
                    "reason": not_enough_data_error_msg,
                    "timestamp": self.g_pool.get_timestamp(),
                    "record": True,
                }
            )
            return

        ref_points_3d_unscaled = np.asarray(ref_points_3d_unscaled)
        pupil0_normals = np.asarray(pupil0_normals)
        pupil1_normals = np.asarray(pupil1_normals)
        initial_translation0, initial_translation1 = np.asarray(self.eye_translations)

        smallest_residual = 1000
        scales = list(np.linspace(0.7, 10, 50))
        for s in scales:
            ref_points_3d = ref_points_3d_unscaled * (1, -1, s)

            # initial_rotation and initial_translation are eye pose in world coordinates
            initial_rotation0 = utils.get_initial_eye_camera_rotation(
                pupil0_normals, ref_points_3d
            )
            initial_rotation1 = utils.get_initial_eye_camera_rotation(
                pupil1_normals, ref_points_3d
            )

            eye0 = SphericalCamera(
                observations=pupil0_normals,
                rotation=initial_rotation0,
                translation=initial_translation0,
                fix_rotation=False,
                fix_translation=True,
            )
            eye1 = SphericalCamera(
                observations=pupil1_normals,
                rotation=initial_rotation1,
                translation=initial_translation1,
                fix_rotation=False,
                fix_translation=True,
            )

            initial_spherical_cameras = eye0, eye1
            initial_gaze_targets = ref_points_3d

            ba = BundleAdjustment(fix_gaze_targets=True)
            success, residual, poses_in_world, gaze_targets_in_world = ba.calculate(
                initial_spherical_cameras, initial_gaze_targets
            )
            if residual <= smallest_residual:
                smallest_residual = residual
                scales[-1] = s

        if not success:
            self.notify_all(
                {
                    "subject": "calibration.failed",
                    "reason": solver_failed_to_converge_error_msg,
                    "timestamp": self.g_pool.get_timestamp(),
                    "record": True,
                }
            )
            logger.error("Calibration solver failed to converge.")
            return

        eye0_pose, eye1_pose = poses_in_world

        sphere_pos0 = matched_data[-1]["pupil"]["sphere"]["center"]
        sphere_pos1 = matched_data[-1]["pupil1"]["sphere"]["center"]
        eye0_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye0_pose, sphere_pos0)
        eye1_cam_pose_in_world = utils.get_eye_cam_pose_in_world(eye1_pose, sphere_pos1)

        observed_normals = [gaze_targets_in_world, eye0.observations, eye1.observations]
        nearest_points = utils.calculate_nearest_points_to_targets(
            observed_normals, [np.zeros(6), *poses_in_world], gaze_targets_in_world
        )
        nearest_points_world, nearest_points_eye0, nearest_points_eye1 = nearest_points

        method = "hmd binocular 3d model"
        ts = g_pool.get_timestamp()
        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.successful",
                "method": method,
                "timestamp": ts,
                "record": True,
            }
        )
        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.calibration_data",
                "timestamp": ts,
                "pupil_list": pupil_list,
                "ref_list": ref_list,
                "calibration_method": method,
                "record": True,
            }
        )

        # this is only used by show calibration. TODO: rewrite show calibration.
        user_calibration_data = {
            "timestamp": ts,
            "pupil_list": pupil_list,
            "ref_list": ref_list,
            "calibration_method": method,
        }
        save_object(
            user_calibration_data,
            os.path.join(g_pool.user_dir, "user_calibration_data"),
        )

        mapper_args = {
            "subject": "start_plugin",
            "name": "Binocular_Vector_Gaze_Mapper",
            "args": {
                "eye_camera_to_world_matrix0": eye0_cam_pose_in_world.tolist(),
                "eye_camera_to_world_matrix1": eye1_cam_pose_in_world.tolist(),
                "cal_points_3d": gaze_targets_in_world.tolist(),
                "cal_ref_points_3d": nearest_points_world.tolist(),
                "cal_gaze_points0_3d": nearest_points_eye0.tolist(),
                "cal_gaze_points1_3d": nearest_points_eye1.tolist(),
                "backproject": hasattr(self.g_pool, "capture"),
            },
        }
        self.g_pool.active_calibration_plugin.notify_all(mapper_args)
