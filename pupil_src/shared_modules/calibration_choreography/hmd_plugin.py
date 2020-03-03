"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import os
import typing as T

import file_methods as fm
from pyglui import ui
from gaze_mapping import GazerHMD3D_v1x

from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyNotification,
    ChoreographyAction,
    ChoreographyMode,
    UnsupportedChoreographyModeError,
    CHOREOGRAPHY_PLUGIN_DONT_REGISTER_LABEL,
)


logger = logging.getLogger(__name__)


class _BaseHMDChoreographyPlugin(CalibrationChoreographyPlugin):

    ### Public

    label = CHOREOGRAPHY_PLUGIN_DONT_REGISTER_LABEL

    is_user_selectable = False

    shows_action_buttons = False

    is_session_persistent = False

    def get_init_dict(self):
        # Disable plugin persistance for HMD choreographies
        raise NotImplementedError(
            "HMD calibration choreography plugin is not persistent"
        )

    ### Public - Plugin

    def init_ui(self):
        desc_text = ui.Info_Text("Calibrate gaze parameters to map onto an HMD.")

        super().init_ui()
        self.menu.append(desc_text)

    def recent_events(self, events):
        if self.is_active:
            self.pupil_list.extend(events["pupil"])

    def on_notify(self, note_dict):
        logger.info(note_dict.get("topic", None) or note_dict.get("subject", None))

        try:
            note = ChoreographyNotification.from_dict(note_dict, allow_extra_keys=True)

            if note.action == ChoreographyAction.SHOULD_START and not self.is_active:
                self._prepare_perform_start_from_notification(note_dict)

            elif note.action == ChoreographyAction.ADD_REF_DATA and self.is_active:
                logger.info(f"ADDING REF DATA, LEN = {len(self.ref_list)}")
                logger.info(f"ADDING REF DATA, IS ACTIVE = {self.is_active}")
                self.ref_list += note_dict["ref_data"]

        except (ValueError, KeyError) as err:
            logger.error(
                f"Notification: {note_dict.keys()} not conform. Raised error {err}"
            )

        else:
            super().on_notify(note_dict)
            logger.info(f"IS ACTIVE = {self.is_active}")

    ### Internal

    @abc.abstractmethod
    def _prepare_perform_start_from_notification(self, note_dict):
        pass


class HMD2DChoreographyPlugin(_BaseHMDChoreographyPlugin):

    ### Public

    label = "HMD 2D Calibration"

    @classmethod
    def supported_gazer_classes(cls):
        return [GazerHMD3D_v1x]

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.__hmd_video_frame_size = None
        self.__outlier_threshold = None

    def on_choreography_successfull(
        self, mode: ChoreographyMode, pupil_list: list, ref_list: list
    ):
        if mode == ChoreographyMode.CALIBRATION:
            self._start_plugin(
                self.selected_gazer_class,
                hmd_video_frame_size=self.__hmd_video_frame_size,
                outlier_threshold=self.__outlier_threshold,
                calib_data={"ref_list": ref_list, "pupil_list": pupil_list},
            )
        elif mode == ChoreographyMode.VALIDATION:
            raise NotImplementedError()
        else:
            raise UnsupportedChoreographyModeError(mode)

    ### Internal

    def _prepare_perform_start_from_notification(self, note_dict):
        self.__hmd_video_frame_size = note_dict["hmd_video_frame_size"]
        self.__outlier_threshold = note_dict["outlier_threshold"]


class HMD3DChoreographyPlugin(_BaseHMDChoreographyPlugin):

    ### Public

    label = "HMD 3D Calibration"

    @classmethod
    def supported_gazer_classes(cls):
        return [GazerHMD3D_v1x]

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.__eye_translations = [0, 0, 0], [0, 0, 0]

    def on_choreography_successfull(
        self, mode: ChoreographyMode, pupil_list: list, ref_list: list
    ):
        if mode == ChoreographyMode.CALIBRATION:
            self._start_plugin(
                self.selected_gazer_class,
                eye_translations=self.__eye_translations,
                calib_data={"ref_list": ref_list, "pupil_list": pupil_list},
            )
        elif mode == ChoreographyMode.VALIDATION:
            raise NotImplementedError()
        else:
            raise UnsupportedChoreographyModeError(mode)

    ### Internal

    def _prepare_perform_start_from_notification(self, note_dict):
        assert len(note_dict["translation_eye0"]) == 3
        assert len(note_dict["translation_eye1"]) == 3
        self.__eye_translations = (
            note_dict["translation_eye0"],
            note_dict["translation_eye1"],
        )
