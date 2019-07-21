"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
from pyglui import ui as gl_ui


class Navigation_Button:
    def __init__(
        self,
        selector: str,
        label_text: str,
        on_click: t.Callable[[], None],
        pupil_icon: chr,
        hotkey: str,
    ):
        self._selector = selector
        self._label_text = label_text
        self._on_click = on_click
        self._pupil_icon = pupil_icon
        self._hotkey = hotkey
        self._ui_button = None

    def add_to_quickbar(self, quickbar):
        on_click = self._on_click
        self._ui_button = gl_ui.Thumb(
            self._selector,
            setter=lambda _: on_click(),
            getter=lambda: False,
            label_font="pupil_icons",
            label=self._pupil_icon,
            hotkey=self._hotkey,
        )
        self._ui_button.status_text = self._label_text
        quickbar.append(self._ui_button)

    def remove_from_quickbar(self, quickbar):
        if self._ui_button:
            quickbar.remove(self._ui_button)
            self._ui_button = None


class Prev_Segment_Button(Navigation_Button):
    def __init__(
        self, on_click: t.Callable[[], None], pupil_icon=chr(0xE045), hotkey="F"
    ):
        super().__init__(
            selector="jump_to_prev_segment",
            label_text="Previous Segment",
            on_click=on_click,
            pupil_icon=pupil_icon,
            hotkey=hotkey,
        )


class Next_Segment_Button(Navigation_Button):
    def __init__(
        self, on_click: t.Callable[[], None], pupil_icon=chr(0xE044), hotkey="f"
    ):
        super().__init__(
            selector="jump_to_next_segment",
            label_text="Next Segment",
            on_click=on_click,
            pupil_icon=pupil_icon,
            hotkey=hotkey,
        )
