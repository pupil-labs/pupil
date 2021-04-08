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
    def ANNOTATION_EVENT_DEFAULT_HOTKEY():
        """Add annotation (default hotkey)

        Software: Capture, Player
        """
        return "x"

    @staticmethod
    def CAMERA_INTRINSIC_ESTIMATOR_COLLECT_NEW_CAPTURE_HOTKEY():
        """Camera intrinsic estimator - collect new

        Software: Capture
        """
        return "i"

    @staticmethod
    def EXPORT_START_PLAYER_HOTKEY():
        """Start export

        Software: Player
        """
        return "e"

    @staticmethod
    def FIXATION_NEXT_PLAYER_HOTKEY():
        """Fixation - show next

        Software: Player
        """
        return "f"

    @staticmethod
    def FIXATION_PREV_PLAYER_HOTKEY():
        """Fixation - show previous

        Software: Player
        """
        return "F"

    @staticmethod
    def GAZE_CALIBRATION_CAPTURE_HOTKEY():
        """Start and stop calibration

        Software: Capture
        """
        return "c"

    @staticmethod
    def GAZE_VALIDATION_CAPTURE_HOTKEY():
        """Start and stop validation

        Software: Capture
        """
        return "t"

    @staticmethod
    def RECORDER_RUNNING_TOGGLE_CAPTURE_HOTKEY():
        """Start and stop recording

        Software: Capture
        """
        return "r"

    @staticmethod
    def SURFACE_TRACKER_ADD_SURFACE_CAPTURE_AND_PLAYER_HOTKEY():
        """Surface tracker - add new surface

        Software: Capture, Player
        """
        return "a"

    @staticmethod
    def SEEK_BAR_MOVE_BACKWARDS_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        """Skip to previous frame

        Software: Player
        Printable: <ARROW LEFT>
        """
        return 263

    @staticmethod
    def SEEK_BAR_MOVE_FORWARDSS_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        """Skip to next frame

        Software: Player
        Printable: <ARROW RIGHT>
        """
        return 262

    @staticmethod
    def SEEK_BAR_PLAY_PAUSE_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        """Skip to next frame

        Software: Player
        Printable: <SPACE>
        """
        return 32


def generate_markdown_hotkey_docs() -> str:
    import pandas as pd

    def generate_row(hotkey_id, hotkey_method):
        hotkey_value = hotkey_method.__get__(Hotkey)()
        hotkey_docsring = hotkey_method.__get__(Hotkey).__doc__
        doc_lines = [l.strip() for l in hotkey_docsring.split("\n") if len(l.strip())]

        if len(doc_lines) > 0:
            hotkey_descr = doc_lines[0]
        else:
            hotkey_descr = ""

        if len(doc_lines) > 1:
            hotkey_meta = dict(
                tuple(map(str.strip, l.split(":"))) for l in doc_lines[1:]
            )
        else:
            hotkey_meta = {}

        hotkey_printable = hotkey_meta.get("Printable")
        hotkey_software = sorted(
            s.strip()
            for s in hotkey_meta.get("Software", "").split(",")
            if len(s.strip()) > 0
        )
        available_in_capture = "Capture" in hotkey_software
        available_in_player = "Player" in hotkey_software
        order = int(available_in_capture) + int(available_in_player) * 10

        emoji_true = ":heavy_check_mark:"
        emoji_false = ":heavy_minus_sign:"

        return {
            "_ID": hotkey_id,
            "_Order": order,
            "Hotkey": f"`{hotkey_printable or hotkey_value}`",
            "Description": hotkey_descr,
            "Pupil Capture": emoji_true if available_in_capture else emoji_false,
            "Pupil Player": emoji_true if available_in_player else emoji_false,
        }

    hotkeys_df = pd.DataFrame(
        [
            generate_row(hotkey_id, hotkey_method)
            for hotkey_id, hotkey_method in vars(Hotkey).items()
            if hotkey_id.endswith("_HOTKEY")
        ]
    )
    hotkeys_df = hotkeys_df.set_index(["Hotkey"])
    hotkeys_df = hotkeys_df.sort_values(by=["_Order", "_ID"])
    hotkeys_df = hotkeys_df[[c for c in hotkeys_df.columns if not c.startswith("_")]]
    return hotkeys_df.to_markdown()


if __name__ == "__main__":
    print(generate_markdown_hotkey_docs())
