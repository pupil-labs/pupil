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
        """Camera intrinsic estimation: Take snapshot of circle pattern

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
        """Fixation: Show next

        Software: Player
        """
        return "f"

    @staticmethod
    def FIXATION_PREV_PLAYER_HOTKEY():
        """Fixation: Show previous

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
        """Surface tracker: Add new surface

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
        """Play and pause video

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

        default_order = int(available_in_capture) + int(available_in_player) * 10
        hotkey_order = hotkey_meta.get("Order", 1000 + default_order)

        return {
            "_ID": hotkey_id,
            "_Order": hotkey_order,
            "_In_Capture": available_in_capture,
            "_In_Player": available_in_player,
            "Keyboard Shortcut": f"`{hotkey_printable or hotkey_value}`",
            "Description": hotkey_descr,
        }

    hotkeys_df = pd.DataFrame(
        [
            generate_row(hotkey_id, hotkey_method)
            for hotkey_id, hotkey_method in vars(Hotkey).items()
            if hotkey_id.endswith("_HOTKEY")
        ]
    )
    hotkeys_df = hotkeys_df.set_index(["Keyboard Shortcut"])
    hotkeys_df = hotkeys_df.sort_values(by=["_Order", "_ID"])

    # Only show columns that don't start with an underscore
    visible_columns = [c for c in hotkeys_df.columns if not c.startswith("_")]

    capture_df = hotkeys_df[hotkeys_df["_In_Capture"] == True]
    player_df = hotkeys_df[hotkeys_df["_In_Player"] == True]

    capture_title_md = "# Pupil Capture"
    capture_table_md = capture_df[visible_columns].to_markdown()

    player_title_md = "# Pupil Player"
    player_table_md = player_df[visible_columns].to_markdown()

    return "\n" + "\n\n".join(
        [capture_title_md, capture_table_md, player_title_md, player_table_md]
    )


if __name__ == "__main__":
    print(generate_markdown_hotkey_docs())
