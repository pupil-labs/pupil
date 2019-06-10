import logging
from pyglui import ui

from pi_preview import GAZE_SENSOR_TYPE, Linked_Device

logger = logging.getLogger(__name__)


class GazeSensor:
    get_data_timeout = 100  # ms

    def __init__(self, network, linked_device):
        self._ndsi_sensor = None
        self.network = network
        self.host_uuid = linked_device.uuid
        self.host_name = linked_device.name
        self.offset = [0, 0]
        if self.is_linked:
            self.activate()

    @property
    def is_linked(self):
        return bool(self.host_uuid)

    @property
    def is_online(self):
        return bool(self._ndsi_sensor)

    @property
    def status(self):
        if not self.is_linked:
            return "Not linked"
        elif not self.is_online:
            return "Not connected"
        else:
            return "Connected"

    @property
    def linked_device(self):
        return Linked_Device(self.host_uuid, self.host_name)

    def unlink(self):
        self.deactivate()
        self.host_uuid = None
        self.network = None

    def activate(self):
        if not self.is_linked:
            logger.error("Cannot activate unlinked sensor")
            return

        try:
            sensor = next(
                s
                for s in self.network.sensors.values()
                if s["sensor_type"] == GAZE_SENSOR_TYPE
                and s["host_uuid"] == self.host_uuid
            )
        except StopIteration:
            logger.debug("Host not found")
            self.deactivate()
            return

        self.host_name = sensor["host_name"]
        self._ndsi_sensor = self.network.sensor(
            sensor["sensor_uuid"], callbacks=(self.on_notification,)
        )
        self._ndsi_sensor.set_control_value("streaming", True)
        self._ndsi_sensor.refresh_controls()
        logger.info("Linked {}".format(self._ndsi_sensor))

    def deactivate(self):
        if self.is_online:
            self._ndsi_sensor.unlink()
            self._ndsi_sensor = None

    def poll_notifications(self):
        if self.is_online:
            while self._ndsi_sensor.has_notifications:
                self._ndsi_sensor.handle_notification()

    def on_notification(self, sensor, event):
        if event["subject"] == "error":
            logger.warning("Error {}".format(event["error_str"]))
            if (
                "control_id" in event
                and event["control_id"] in self._ndsi_sensor.controls
            ):
                logger.info(str(self._ndsi_sensor.controls[event["control_id"]]))

    def fetch_data(self):
        if self.is_online:
            return [
                self._make_gaze_pos(x, y, ts)
                for (x, y, ts) in self._ndsi_sensor.fetch_data()
            ]
        return []

    @staticmethod
    def _make_gaze_pos(x, y, ts, frame_size_x=1080, frame_size_y=1080):
        return {
            "topic": "gaze.pi",
            "norm_pos": [x / frame_size_x, 1.0 - y / frame_size_y],
            "timestamp": ts,
            "confidence": 1.0,
        }

    def add_ui_elements(self, menu):
        menu.append(
            ui.Text_Input("status", self, label="Status", setter=lambda _: None)
        )
        if self.is_linked:
            menu.append(
                ui.Text_Input(
                    "host_name", self, label="Linked device", setter=lambda _: None
                )
            )

