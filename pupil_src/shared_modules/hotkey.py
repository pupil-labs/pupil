"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2021 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class Hotkey:
    """"""

    @staticmethod
    def CAMERA_INTRINSIC_ESTIMATOR_COLLECT_NEW_CAPTURE_HOTKEY():
        return "i"

    @staticmethod
    def EXPORT_START_PLAYER_HOTKEY():
        return "e"

    @staticmethod
    def FIXATION_NEXT_PLAYER_HOTKEY():
        return "f"

    @staticmethod
    def FIXATION_PREV_PLAYER_HOTKEY():
        return "F"

    @staticmethod
    def GAZE_CALIBRATION_CAPTURE_HOTKEY():
        return "c"

    @staticmethod
    def GAZE_VALIDATION_CAPTURE_HOTKEY():
        return "t"

    @staticmethod
    def RECORDER_RUNNING_TOGGLE_CAPTURE_HOTKEY():
        return "r"

    @staticmethod
    def SURFACE_TRACKER_ADD_SURFACE_CAPTURE_AND_PLAYER_HOTKEY():
        return "a"

    @staticmethod
    def SEEK_BAR_MOVE_BACKWARDS_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        return 263

    @staticmethod
    def SEEK_BAR_MOVE_FORWARDSS_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        return 262

    @staticmethod
    def SEEK_BAR_PLAY_PAUSE_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        return 32
