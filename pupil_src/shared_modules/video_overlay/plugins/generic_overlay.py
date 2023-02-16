"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import os

from observable import Observable
from plugin import Plugin
from video_capture.utils import VIDEO_EXTS
from video_overlay.controllers.overlay_manager import OverlayManager
from video_overlay.models.config import Configuration
from video_overlay.ui.interactions import current_mouse_pos
from video_overlay.ui.management import UIManagementGeneric


class Video_Overlay(Observable, Plugin):
    icon_chr = "O"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.manager = OverlayManager(g_pool.rec_dir)

    def get_init_dict(self):
        # Save current settings to disk, ensures that the World Video Exporter
        # loads the most recent settings.
        self.manager.save_to_disk()
        return super().get_init_dict()

    def recent_events(self, events):
        if "frame" in events:
            frame = events["frame"]
            for overlay in self.manager.overlays:
                overlay.draw_on_frame(frame)

    def on_drop(self, paths):
        multi_drop_offset = 10
        inital_drop = current_mouse_pos(
            self.g_pool.main_window,
            self.g_pool.camera_render_size,
            self.g_pool.capture.frame_size,
        )
        valid_paths = [p for p in paths if self.valid_path(p)]
        for video_path in valid_paths:
            self._add_overlay_to_storage(video_path, inital_drop)
            inital_drop = (
                inital_drop[0] + multi_drop_offset,
                inital_drop[1] + multi_drop_offset,
            )
        # event only consumed if at least one valid file was present
        return bool(valid_paths)

    @staticmethod
    def valid_path(path):
        # splitext()[1] always starts with `.`
        ext = os.path.splitext(path)[1][1:]
        return ext in VIDEO_EXTS

    def _add_overlay_to_storage(self, video_path, initial_pos=(0, 0)):
        config = Configuration(
            video_path, origin_x=initial_pos[0], origin_y=initial_pos[1]
        )
        self.manager.add(config)
        self.manager.save_to_disk()
        self._overlay_added_to_storage(self.manager.most_recent)

    def _overlay_added_to_storage(self, overlay):
        pass  # observed to create menus and draggables

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Generic Video Overlays"
        self.ui = UIManagementGeneric(self, self.menu, self.manager.overlays)
        self.ui.add_observer("remove_overlay", self.manager.remove_overlay)

    def deinit_ui(self):
        self.ui.teardown()
        self.remove_menu()
