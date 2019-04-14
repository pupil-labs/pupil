from plugin import Plugin

from video_overlay.controllers.overlay import Controller as OverlayController


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
