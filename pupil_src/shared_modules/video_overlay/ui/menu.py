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
import os
import weakref

from observable import Observable
from pyglui import ui


def make_scale_slider(config):
    return ui.Slider(
        "value",
        config.scale,
        label="Scale",
        min=config.scale.constraint.low,
        max=config.scale.constraint.high,
        step=0.05,
    )


def make_alpha_slider(config):
    return ui.Slider(
        "value",
        config.alpha,
        label="Opacity",
        min=config.alpha.constraint.low,
        max=config.alpha.constraint.high,
        step=0.05,
    )


def make_hflip_switch(config):
    return ui.Switch("value", config.hflip, label="Flip horizontally")


def make_vflip_switch(config):
    return ui.Switch("value", config.vflip, label="Flip vertically")


class OverlayMenuRenderer(Observable, abc.ABC):
    def __init__(self, overlay):
        self.overlay = weakref.ref(overlay)
        video_basename = os.path.basename(self.overlay().config.video_path)
        self.menu = ui.Growing_Menu(video_basename)
        self.menu.collapsed = True
        self.update_menu()

    def update_menu(self):
        if self.overlay().valid_video_loaded:
            self.menu[:] = self._generic_overlay_elements()
        else:
            self.menu[:] = self._not_valid_video_elements()
        self._append_remove_button()

    def _append_remove_button(self):
        pass  # do not show remove button by default

    @abc.abstractmethod
    def _generic_overlay_elements(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _not_valid_video_elements(self):
        raise NotImplementedError


class GenericOverlayMenuRenderer(OverlayMenuRenderer):
    def _append_remove_button(self):
        self.menu.append(ui.Button("Remove overlay", self._remove))

    def _remove(self):
        self.remove_button_clicked(self.overlay())

    def remove_button_clicked(self, overlay):
        pass  # observable

    def _generic_overlay_elements(self):
        config = self.overlay().config
        return (
            ui.Info_Text(f"Loaded video: {config.video_path}"),
            make_scale_slider(config),
            make_alpha_slider(config),
            make_hflip_switch(config),
            make_vflip_switch(config),
        )

    def _not_valid_video_elements(self):
        video_path = self.overlay().config.video_path
        return (
            ui.Info_Text(f"No valid overlay video found at {video_path}"),
            ui.Info_Text(
                "Valid overlay videos conform to the Pupil data format and "
                "their timestamps are in sync with the opened recording."
            ),
        )


class EyesOverlayMenuRenderer(OverlayMenuRenderer):
    def _generic_overlay_elements(self):
        config = self.overlay().config
        return (make_hflip_switch(config), make_vflip_switch(config))

    def _not_valid_video_elements(self):
        video_path = self.overlay().config.video_path
        video_name = os.path.basename(video_path)
        return (ui.Info_Text(f"{video_name} was not recorded or cannot be found."),)
