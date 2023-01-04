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
import logging
import os
import typing as T

import file_methods as fm
from gaze_mapping import GazerHMD3D

from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyAction,
    ChoreographyMode,
    ChoreographyNotification,
    UnsupportedChoreographyModeError,
)

logger = logging.getLogger(__name__)


class _BaseHMDChoreographyPlugin(CalibrationChoreographyPlugin):

    ### Public

    @classmethod
    def should_register(cls) -> bool:
        if cls.__name__ == "_BaseHMDChoreographyPlugin":
            return False
        return super().should_register()

    is_user_selectable = False

    shows_action_buttons = False

    is_session_persistent = False

    def get_init_dict(self):
        # Disable plugin persistance for HMD choreographies
        raise NotImplementedError(
            "HMD calibration choreography plugin is not persistent"
        )

    ### Public - Plugin

    def __init__(self, *args, **kwargs):
        type(self).is_user_selectable = True
        super().__init__(*args, **kwargs)

    def cleanup(self):
        type(self).is_user_selectable = False
        super().cleanup()

    @classmethod
    def _choreography_description_text(cls) -> str:
        return "Calibrate gaze parameters to map onto an HMD."

    def recent_events(self, events):
        super().recent_events(events)

        if self.is_active:
            self.pupil_list.extend(events["pupil"])

    def on_notify(self, note_dict):
        try:
            note = ChoreographyNotification.from_dict(note_dict)
        except ValueError:
            return  # Unknown/unexpected notification, not handling it
        else:
            if note.action == ChoreographyAction.SHOULD_START and not self.is_active:
                try:
                    self._prepare_perform_start_from_notification(note_dict)
                except KeyError as err:
                    logger.error(f"Calibration cannot be started without {err}")
                    return

            elif note.action == ChoreographyAction.ADD_REF_DATA and self.is_active:
                self.ref_list += note_dict["ref_data"]

            super().on_notify(note_dict)

    ### Internal

    @abc.abstractmethod
    def _prepare_perform_start_from_notification(self, note_dict):
        pass


class HMD3DChoreographyPlugin(_BaseHMDChoreographyPlugin):

    ### Public

    label = "HMD 3D Calibration"

    @classmethod
    def supported_gazer_classes(cls):
        return [GazerHMD3D]

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool, **kwargs)
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
