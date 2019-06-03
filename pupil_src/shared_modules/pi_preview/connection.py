import logging
import ndsi
from pyglui import ui

try:
    from ndsi import __version__

    assert __version__ >= "0.5"
    from ndsi import __protocol_version__
except (ImportError, AssertionError):
    raise Exception("pyndsi version is to old. Please upgrade") from None

logger = logging.getLogger(__name__)


class Connection:
    def __init__(self, update_ui_cb):
        self.update_ui_cb = update_ui_cb
        self.network = ndsi.Network(callbacks=(self.on_event,))
        self.network.start()
        self.selected_sensor_uuid = None
        self.sensor = None
        self.get_data_timeout = 100  # ms

    def close(self):
        self.network.stop()

    def update(self):
        self.poll_events()

        if self.sensor:
            self.poll_notifications()

        return self.fetch_data()

    def fetch_data(self):
        if self.sensor:
            return [
                self._make_gaze_pos(x, y, ts) for (x, y, ts) in self.sensor.fetch_data()
            ]
        return []

    @staticmethod
    def _make_gaze_pos(x, y, ts, frame_size_x=1080, frame_size_y=1080):
        return {
            "topic": "gaze.pi",
            "norm_pos": (x / frame_size_x, 1.0 - y / frame_size_y),
            "timestamp": ts,
            "confidence": 1.0,
        }

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def poll_notifications(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def on_event(self, caller, event):
        if event["subject"] == "detach":
            logger.debug("detached: %s" % event)
            sensors = [s for s in self.network.sensors.values()]
            if self.selected_sensor_uuid == event["sensor_uuid"]:
                if sensors:
                    self._select_sensor(sensors[0]["sensor_uuid"])
                else:
                    self._select_sensor(None)

        elif event["subject"] == "attach" and event["sensor_type"] == "gaze":
            if not self.selected_sensor_uuid:
                self._select_sensor(event["sensor_uuid"])
            else:
                self.update_ui_cb()

    def on_notification(self, sensor, event):
        if event["subject"] == "error":
            logger.warning("Error {}".format(event["error_str"]))
            if "control_id" in event and event["control_id"] in self.sensor.controls:
                logger.info(str(self.sensor.controls[event["control_id"]]))

    def add_ui_elements(self, menu):
        def host_selection_list():
            devices = {
                s["sensor_uuid"]: s["host_name"]  # removes duplicates
                for s in self.network.sensors.values()
                if s["sensor_type"] == "gaze"
            }
            devices = [pair for pair in devices.items()]  # create tuples
            # split tuples into 2 lists
            return zip(*(devices or [(None, "No hosts found")]))

        host_sel, host_sel_labels = host_selection_list()
        menu.append(
            ui.Selector(
                "selected_sensor_uuid",
                self,
                selection=host_sel,
                labels=host_sel_labels,
                setter=self._select_sensor,
                label="Remote host",
            )
        )

    def _select_sensor(self, sensor_uuid):
        if self.selected_sensor_uuid == sensor_uuid:
            return

        self.selected_sensor_uuid = sensor_uuid
        self.update_ui_cb()

        if sensor_uuid:
            self._activate_sensor(sensor_uuid)
        else:
            self._deactivate_sensor()

    def _activate_sensor(self, sensor_uuid):
        self.sensor = self.network.sensor(
            sensor_uuid, callbacks=(self.on_notification,)
        )
        self.sensor.set_control_value("streaming", True)
        self.sensor.refresh_controls()
        logger.info("Activated {}".format(self.sensor))

    def _deactivate_sensor(self):
        if self.sensor:
            logger.warning("Deactivated {}".format(self.sensor))
            self.sensor.unlink()
        self.sensor = None
