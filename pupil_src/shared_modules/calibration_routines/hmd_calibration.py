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

from pyglui import ui

import audio
import file_methods as fm
from calibration_routines import data_processing
from calibration_routines.calibration_plugin_base import Calibration_Plugin
from calibration_routines.finish_calibration import (
    create_converge_error_msg,
    create_not_enough_data_error_msg,
)
from calibration_routines.optimization_calibration import calibration_methods

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
            logger.error(f"Notification: {notification} not conform. Raised error {e}")

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
        g_pool = self.g_pool

        extracted_data = data_processing.get_data_for_calibration_hmd(
            pupil_list, ref_list, mode="2d"
        )
        if not extracted_data:
            self.notify_all(create_not_enough_data_error_msg(g_pool))
            return

        method, result = calibration_methods.calibrate_2d_hmd(
            self.hmd_video_frame_size, *extracted_data
        )
        if result is None:
            self.notify_all(create_converge_error_msg(g_pool))
            return

        ts = g_pool.get_timestamp()

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
        self.notify_all(result)

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

        extracted_data = data_processing.get_data_for_calibration_hmd(
            pupil_list, ref_list, mode="3d"
        )
        if not extracted_data:
            self.notify_all(create_not_enough_data_error_msg(g_pool))
            return

        method, result = calibration_methods.calibrate_3d_hmd(
            *extracted_data, self.eye_translations
        )
        if result is None:
            self.notify_all(create_converge_error_msg(g_pool))
            return

        ts = g_pool.get_timestamp()

        # Announce success
        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.successful",
                "method": method,
                "timestamp": ts,
                "record": True,
            }
        )

        # Announce calibration data
        # this is only used by show calibration. TODO: rewrite show calibration.
        user_calibration_data = {
            "timestamp": ts,
            "pupil_list": pupil_list,
            "ref_list": ref_list,
            "calibration_method": method,
        }
        fm.save_object(
            user_calibration_data,
            os.path.join(g_pool.user_dir, "user_calibration_data"),
        )
        g_pool.active_calibration_plugin.notify_all(
            {
                "subject": "calibration.calibration_data",
                "record": True,
                **user_calibration_data,
            }
        )

        # Start mapper
        result["args"]["backproject"] = hasattr(g_pool, "capture")
        self.g_pool.active_calibration_plugin.notify_all(result)
