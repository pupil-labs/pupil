import logging
import ndsi
from pyglui import ui

from pi_preview import GAZE_SENSOR_TYPE, Linked_Device
from pi_preview.sensor import GazeSensor

logger = logging.getLogger(__name__)


class Connection:
    def __init__(self, linked_device, update_ui_cb, activate_ndsi_backend_cb):
        self.update_ui_cb = update_ui_cb
        self.activate_ndsi_backend_cb = activate_ndsi_backend_cb
        self.network = ndsi.Network(callbacks=(self.on_event,))
        self.network.start()
        self.sensor = GazeSensor(self.network, linked_device)

    def close(self):
        self.sensor.unlink()
        self.network.stop()

    def fetch_data(self):
        self.poll_events()
        self.sensor.poll_notifications()
        return self.sensor.fetch_data()

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def on_event(self, caller, event):
        if (
            event["subject"] == "attach"
            and event["sensor_type"] == GAZE_SENSOR_TYPE
            and event["host_uuid"] == self.sensor.host_uuid
        ):
            self.sensor.activate()
            self.update_ui_cb()

        if event["subject"] == "detach" and event["host_uuid"] == self.sensor.host_uuid:
            self.sensor.deactivate()

    def add_ui_elements(self, menu):
        def host_selection_list():
            default = (None, "Select to link")
            devices = {
                s["host_uuid"]: s["host_name"]  # removes duplicates
                for s in self.network.sensors.values()
                if s["sensor_type"] == GAZE_SENSOR_TYPE
            }
            devices = [default, *devices.items()]  # create list of tuples
            # split tuples into 2 lists
            return zip(*devices)

        menu.append(
            ui.Selector(
                "available_sensors",
                getter=lambda: None,
                selection_getter=host_selection_list,
                setter=self._select_sensor,
                label="Link",
            )
        )
        self.sensor.add_ui_elements(menu)

    def _select_sensor(self, host_uuid):
        if not host_uuid:
            return
        elif self.sensor.is_linked and self.sensor.host_uuid == host_uuid:
            logger.info("Host already linked")
            return
        elif self.sensor.is_linked:
            self.sensor.unlink()

        self.sensor = GazeSensor(self.network, Linked_Device(uuid=host_uuid, name=None))
        self.update_ui_cb()
        self.activate_ndsi_backend_cb(host_uuid)
