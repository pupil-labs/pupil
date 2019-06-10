import logging

import glfw
from pyglui import ui

from plugin import Plugin

from pi_preview import Linked_Device
from pi_preview.connection import Connection
from pi_preview.filter import TemporalFilter, OffsetFilter

logger = logging.getLogger(__name__)

IMG_WIDTH = 1088
IMG_HEIGHT = 1080


class PI_Preview(Plugin):
    icon_chr = "PI"
    order = 0.02  # ensures init after all plugins

    def __init__(
        self,
        g_pool,
        linked_device=...,
        temporal_filter_config=...,
        offset_filter_config=...,
    ):
        super().__init__(g_pool)

        if linked_device is ...:
            linked_device = Linked_Device(None, None)
        else:
            linked_device = Linked_Device(*linked_device)

        if temporal_filter_config is ...:
            temporal_filter_config = {"enabled": True}
        if offset_filter_config is ...:
            offset_filter_config = {"enabled": False}

        self.temporal_filter = TemporalFilter(**temporal_filter_config)
        self.offset_filter = OffsetFilter(**offset_filter_config)

        self.connection = Connection(
            linked_device,
            update_ui_cb=self.update_ndsi_menu,
            activate_ndsi_backend_cb=self.activate_ndsi_backend,
        )
        self._num_prefix_elements = 0
        self.last_click = None
        self.default_config()

    def on_click(self, pos, button, action):
        if action == glfw.GLFW_PRESS:
            self.last_click = pos[0] / IMG_WIDTH, (1.0 - pos[1] / IMG_HEIGHT)

    def recent_events(self, events):
        gaze = self.connection.fetch_data()
        for datum in gaze:
            self.temporal_filter.apply(datum)

        if self.last_click and gaze:
            self.offset_filter.offset = (
                self.last_click[0] - gaze[-1]["norm_pos"][0],
                self.last_click[1] - gaze[-1]["norm_pos"][1],
            )
            self.last_click = None

        for datum in gaze:
            self.offset_filter.apply(datum)

        if gaze:
            if "gaze" not in events:
                events["gaze"] = gaze
            else:
                events["gaze"].extend(gaze)

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Pupil Invisible Preview"
        self.menu.append(ui.Info_Text("Filter settings"))
        self.temporal_filter.add_ui_elements(self.menu)
        self.offset_filter.add_ui_elements(self.menu)
        self.menu.append(ui.Info_Text("Connection settings"))
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

    def get_init_dict(self):
        return {
            "linked_device": self.connection.sensor.linked_device,
            "temporal_filter_config": self.temporal_filter.get_init_dict(),
            "offset_filter_config": self.offset_filter.get_init_dict(),
        }

    def activate_ndsi_backend(self, host_uuid):
        self.notify_all(
            {"subject": "backend.auto_select_manager", "name": "NDSI_Manager"}
        )
        self.notify_all(
            {
                "subject": "backend.ndsi_do_select_host",
                "target_host": host_uuid,
                "delay": 0.4,
            }
        )
        self.notify_all({"subject": "pi_preview.focus_menu", "delay": 1.0})
        self.notify_all({"subject": "world_process.adapt_window_size", "delay": 5.0})

    def default_config(self):
        self.notify_all({"subject": "eye_process.should_stop", "eye_id": 0})
        self.notify_all({"subject": "eye_process.should_stop", "eye_id": 1})
        self.notify_all({"subject": "start_plugin", "name": "HMD_Calibration"})
        self.notify_all({"subject": "stop_plugin", "name": "Recorder"})
        self.notify_all({"subject": "pi_preview.focus_menu"})

    def on_notify(self, notification):
        if notification["subject"] == "pi_preview.focus_menu":
            self._toggle_menu(False)

    def _toggle_menu(self, collapsed):
        # This is the menu toggle logic.
        # Only one menu can be open.
        # If no menu is open the menu_bar should collapse.
        self.g_pool.menubar.collapsed = collapsed
        for m in self.g_pool.menubar.elements:
            m.collapsed = True
        self.menu.collapsed = collapsed
