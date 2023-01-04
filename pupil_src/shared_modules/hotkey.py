"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


class Hotkey:
    """"""

    @staticmethod
    def ANNOTATION_EVENT_DEFAULT_HOTKEY():
        """Add annotation (default keyboard shortcut)

        Capture Order: 40
        Player Order: 50
        """
        return "x"

    @staticmethod
    def CAMERA_INTRINSIC_ESTIMATOR_COLLECT_NEW_CAPTURE_HOTKEY():
        """Camera intrinsic estimation: Take snapshot of circle pattern

        Capture Order: 50
        """
        return "i"

    @staticmethod
    def EXPORT_START_PLAYER_HOTKEY():
        """Start export

        Player Order: 30
        """
        return "e"

    @staticmethod
    def FIXATION_NEXT_PLAYER_HOTKEY():
        """Fixation: Show next

        Player Order: 60
        """
        return "f"

    @staticmethod
    def FIXATION_PREV_PLAYER_HOTKEY():
        """Fixation: Show previous

        Player Order: 61
        """
        return "F"

    @staticmethod
    def GAZE_CALIBRATION_CAPTURE_HOTKEY():
        """Start and stop calibration

        Capture Order: 20
        """
        return "c"

    @staticmethod
    def GAZE_VALIDATION_CAPTURE_HOTKEY():
        """Start and stop validation

        Capture Order: 21
        """
        return "t"

    @staticmethod
    def RECORDER_RUNNING_TOGGLE_CAPTURE_HOTKEY():
        """Start and stop recording

        Capture Order: 10
        """
        return "r"

    @staticmethod
    def SURFACE_TRACKER_ADD_SURFACE_CAPTURE_AND_PLAYER_HOTKEY():
        """Surface tracker: Add new surface

        Capture Order: 30
        Player Order: 40
        """
        return "a"

    @staticmethod
    def SEEK_BAR_MOVE_BACKWARDS_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        """Step to previous frame\\* / Decrease playback speed\\*\\*

        Printable: <arrow left>
        Player Order: 20
        """
        return 263

    @staticmethod
    def SEEK_BAR_MOVE_FORWARDS_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        """Step to next frame\\* / Increase playback speed\\*\\*

        Printable: <arrow right>
        Player Order: 21
        """
        return 262

    @staticmethod
    def SEEK_BAR_PLAY_PAUSE_PLAYER_HOTKEY():
        # This is only implicitly used by pyglui.ui.Seek_Bar
        """Play and pause video

        Printable: <space>
        Player Order: 10
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
        hotkey_order_in_capture = hotkey_meta.get("Capture Order", None)
        hotkey_order_in_player = hotkey_meta.get("Player Order", None)

        return {
            "_ID": hotkey_id,
            "_Order_In_Capture": hotkey_order_in_capture,
            "_Order_In_Player": hotkey_order_in_player,
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

    # Only show columns that don't start with an underscore
    visible_columns = [c for c in hotkeys_df.columns if not c.startswith("_")]

    capture_df = hotkeys_df[hotkeys_df["_Order_In_Capture"].notnull()]
    capture_df = capture_df.sort_values(by=["_Order_In_Capture"])

    player_df = hotkeys_df[hotkeys_df["_Order_In_Player"].notnull()]
    player_df = player_df.sort_values(by=["_Order_In_Player"])

    capture_title_md = "# Pupil Capture"
    capture_table_md = capture_df[visible_columns].to_markdown()

    player_title_md = "# Pupil Player"
    player_table_md = player_df[visible_columns].to_markdown()

    player_footnote = "\\* While paused\n\\* During playback"

    return "\n" + "\n\n".join(
        [
            capture_title_md,
            capture_table_md,
            player_title_md,
            player_table_md,
            player_footnote,
        ]
    )


if __name__ == "__main__":
    print(generate_markdown_hotkey_docs())
