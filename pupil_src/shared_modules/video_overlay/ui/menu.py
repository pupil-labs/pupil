import os
import weakref

from pyglui import ui

from observable import Observable


def not_valid_video_elements(video_path):
    return (
        ui.Info_Text("No valid overlay video found at {}".format(video_path)),
        ui.Info_Text(
            "Valid overlay videos conform to the Pupil data format and "
            "their timestamps are in sync with the opened recording."
        ),
    )


def generic_overlay_elements(config):
    return (
        ui.Info_Text("Loaded video: {}".format(config.video_path)),
        ui.Slider(
            "value",
            config.scale,
            label="Scale",
            min=config.scale.constraint.low,
            max=config.scale.constraint.high,
            step=0.05,
        ),
        ui.Slider(
            "value",
            config.alpha,
            label="Transparency",
            min=config.alpha.constraint.low,
            max=config.alpha.constraint.high,
            step=0.05,
        ),
        ui.Switch("value", config.hflip, label="Flip horizontally"),
        ui.Switch("value", config.vflip, label="Flip vertically"),
    )


class GenericOverlayMenuRenderer(Observable):
    def __init__(self, overlay):
        self.overlay = weakref.ref(overlay)
        video_basename = os.path.basename(self.overlay().config.video_path)
        self.menu = ui.Growing_Menu(video_basename)
        self.menu.collapsed = True
        self.update_menu()

    def update_menu(self):
        if self.overlay().valid_video_loaded:
            self.menu[:] = generic_overlay_elements(self.overlay().config)
        else:
            self.menu[:] = not_valid_video_elements(self.overlay().config.video_path)
        self._append_remove_button()

    def _append_remove_button(self):
        self.menu.append(ui.Button("Remove overlay", self._remove))

    def _remove(self):
        self.remove_button_clicked(self.overlay())

    def remove_button_clicked(self, overlay):
        pass  # observable
