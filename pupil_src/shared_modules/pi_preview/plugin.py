import logging

import glfw
from pyglui import ui

from plugin import Plugin

from pi_preview.connection import Connection

logger = logging.getLogger(__name__)

IMG_SIZE = 1080


class PI_Preview(Plugin):
    icon_chr = "PI"
    order = 0.02  # ensures init after all plugins

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.connection = Connection(update_ui_cb=self.update_ndsi_menu)
        self._num_prefix_elements = 0
        self.offset = [0, 0]
        self.last_click = None

    def on_click(self, pos, button, action):
        if action == glfw.GLFW_PRESS:
            self.last_click = pos[0] / IMG_SIZE, (IMG_SIZE - pos[1]) / IMG_SIZE

    def recent_events(self, events):
        gaze = self.connection.update()
        if gaze and self.last_click:
            self.offset = (
                self.last_click[0] - gaze[0]["norm_pos"][0],
                self.last_click[1] - gaze[0]["norm_pos"][1],
            )
            self.last_click = None

        for g in gaze:
            g["norm_pos"][0] += self.offset[0]
            g["norm_pos"][1] += self.offset[1]

        if gaze:
            if "gaze" not in events:
                events["gaze"] = gaze
            else:
                events["gaze"].extend(gaze)

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Pupil Invisible Preview"
        self._num_prefix_elements = len(self.menu)
        self.update_ndsi_menu()

    def deinit_ui(self):
        self.remove_menu()

    def update_ndsi_menu(self):
        del self.menu[self._num_prefix_elements :]
        self.connection.add_ui_elements(self.menu)

    def cleanup(self):
        self.connection.close()
        self.connection = None

