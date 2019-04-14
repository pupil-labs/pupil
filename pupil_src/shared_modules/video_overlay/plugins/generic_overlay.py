import os

from plugin import Plugin

from video_overlay.controllers.overlay import Controller as OverlayController
from video_overlay.ui.menu import generic_overlay_elements, no_valid_video_elements


class Vis_Generic_Video_Overlay(Plugin):
    uniqueness = "not_unique"

    def __init__(self, g_pool, video_path=None, config=None):
        super().__init__(g_pool)
        config = config or {"scale": 0.5, "alpha": 0.9, "hflip": False, "vflip": False}
        self.controller = OverlayController(video_path, config)

    def get_init_dict(self):
        return {
            "video_path": self.controller.video_path,
            "config": self.controller.config.get_init_dict(),
        }

    def recent_events(self, events):
        if "frame" in events:
            frame = events["frame"]
            self.controller.draw_on_frame(frame)

    def on_drop(self, paths):
        remaining_paths = paths.copy()
        while remaining_paths and not self.controller.valid_video_loaded:
            video_path = remaining_paths.pop(0)
            if self.controller.attempt_to_load_video(video_path):
                return True  # event consumed
        return False  # event not consumed

    def init_ui(self):
        self.add_menu()
        self._refresh_menu()
        self.controller.add_observer("attempt_to_load_video", self._refresh_menu)

    def deinit_ui(self):
        self.controller.remove_observer("attempt_to_load_video", self._refresh_menu)
        self.remove_menu()

    def _refresh_menu(self, *args, **kwargs):

        if self.controller.valid_video_loaded:
            menu_elements = generic_overlay_elements(
                self.controller.video_path, self.controller.config
            )
            icon_chr = "O"
            title = "Video Overlay: {}".format(
                os.path.basename(self.controller.video_path)
            )
        else:
            menu_elements = no_valid_video_elements()
            icon_chr = "!"
            title = "Video Overlay: No valid video loaded"
        # first element corresponds to `Close` button, added in add_menu()
        self.menu[1:] = menu_elements
        self.menu_icon.label = icon_chr
        self.menu.label = title
