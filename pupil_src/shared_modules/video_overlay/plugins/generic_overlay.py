import collections
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
        self.refresh_menu()
        self.controller.add_observer("attempt_to_load_video", self.refresh_menu)

    def deinit_ui(self):
        self.controller.remove_observer("attempt_to_load_video", self.refresh_menu)
        self.remove_menu()

    def refresh_menu(self, *args, **kwargs):
        ui_setup = self._ui_setup()
        # first element corresponds to `Close` button, added in add_menu()
        self.menu_icon.label = ui_setup.icon
        self.menu.label = ui_setup.title
        self.menu[1:] = ui_setup.menu_elements

    def _ui_setup(self):
        return (
            self._ui_setup_if_valid()
            if self.controller.valid_video_loaded
            else self._ui_setup_if_not_valid()
        )

    def _ui_setup_if_valid(self):
        video_basename = os.path.basename(self.controller.video_path)
        menu_elements = generic_overlay_elements(
            self.controller.video_path, self.controller.config
        )
        return UISetup(
            icon="O",
            title="Video Overlay: {}".format(video_basename),
            menu_elements=menu_elements,
        )

    def _ui_setup_if_not_valid(self):
        return UISetup(
            icon="!",
            title="Video Overlay: No valid video loaded",
            menu_elements=no_valid_video_elements(),
        )


UISetup = collections.namedtuple("UISetup", ("icon", "title", "menu_elements"))
