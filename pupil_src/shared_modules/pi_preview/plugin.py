import logging

from pyglui import ui

from plugin import Plugin

from pi_preview.connection import Connection

logger = logging.getLogger(__name__)


class PI_Preview(Plugin):
    icon_chr = "PI"
    order = 0.02  # ensures init after all plugins

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.connection = Connection(update_ui_cb=self.update_ndsi_menu)
        self._num_prefix_elements = 0

    def recent_events(self, events):
        self.connection.update()
        frame = events.get("frame")

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

