"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import time

import ndsi
import os_utils
from camera_models import Camera_Model
from pyglui import ui
from version_utils import parse_version

from .base_backend import Base_Manager, Base_Source, SourceInfo

try:
    from ndsi import __version__

    assert parse_version(__version__) >= parse_version("1.3")
    from ndsi import __protocol_version__
except (ImportError, AssertionError):
    raise Exception("pyndsi version is too old. Please upgrade!") from None

os_utils.patch_pyre_zhelper_cdll()
logger = logging.getLogger(__name__)

# Suppress pyre debug logs (except beacon)
logger.debug("Suppressing pyre debug logs (except zbeacon)")
logging.getLogger("pyre").setLevel(logging.WARNING)
logging.getLogger("pyre.zbeacon").setLevel(logging.DEBUG)


class NDSI_Source(Base_Source):
    """Pupil Mobile video source

    Attributes:
        get_frame_timeout (float): Maximal waiting time for next frame
        sensor (ndsi.Sensor): NDSI sensor backend
    """

    def __init__(
        self,
        g_pool,
        frame_size,
        frame_rate,
        source_id=None,
        host_name=None,
        sensor_name=None,
        *args,
        **kwargs,
    ):
        super().__init__(g_pool, *args, **kwargs)
        self.sensor = None
        self._source_id = source_id
        self._sensor_name = sensor_name
        self._host_name = host_name
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self.ui_initialized = False
        self.control_id_ui_mapping = {}
        self.get_frame_timeout = 100  # ms
        self.ghost_mode_timeout = 10  # sec
        self._initial_refresh = True
        self.last_update = self.g_pool.get_timestamp()

        manager = next((p for p in g_pool.plugins if isinstance(p, NDSI_Manager)), None)
        if not manager:
            logger.error("Error connecting to Pupil Mobile: NDSI Manager not found!")
            return

        network = manager.network
        if not network:
            logger.error("Error connecting to Pupil Mobile: No NDSI network!")
            return

        self.recover(network)

        if not self.sensor or not self.sensor.supports_data_subscription:
            logger.error("Could not connect to device! No images will be supplied.")
            self.cleanup()

        logger.warning(
            "Make sure to enable the Time_Sync plugin for recording with Pupil Mobile!"
        )

    def recover(self, network):
        logger.debug(
            "Trying to recover with %s, %s, %s"
            % (self._source_id, self._sensor_name, self._host_name)
        )
        if self._source_id:
            try:
                # uuid given
                self.sensor = network.sensor(
                    self._source_id, callbacks=(self.on_notification,)
                )
            except ValueError:
                pass

        if self.online:
            self._sensor_name = self.sensor.name
            self._host_name = self.sensor.host_name
            return
        if self._host_name and self._sensor_name:
            for sensor in network.sensors.values():
                if (
                    sensor["host_name"] == self._host_name
                    and sensor["sensor_name"] == self._sensor_name
                ):
                    self.sensor = network.sensor(
                        sensor["sensor_uuid"], callbacks=(self.on_notification,)
                    )
                    if self.online:
                        self._sensor_name = self.sensor.name
                        self._host_name = self.sensor.host_name
                        break
        else:
            for s_id in network.sensors:
                self.sensor = network.sensor(s_id, callbacks=(self.on_notification,))
                if self.online:
                    self._sensor_name = self.sensor.name
                    self._host_name = self.sensor.host_name
                    break

    @property
    def name(self):
        return f"{self._sensor_name}"

    @property
    def host(self):
        return f"{self._host_name}"

    @property
    def online(self):
        return bool(self.sensor)

    def poll_notifications(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def recent_events(self, events):
        if self.online:
            self.poll_notifications()
            try:
                frame = self.sensor.get_newest_data_frame(
                    timeout=self.get_frame_timeout
                )
            except ndsi.StreamError:
                frame = None
            except ndsi.sensor.NotDataSubSupportedError:
                # NOTE: This (most likely) is a race-condition in NDSI initialization
                # that is waiting to be fixed for Pupil Mobile. It happens rarely and
                # can be solved by simply reconnecting the headset to the mobile phone.
                # Preventing traceback logfloods here and displaying more helpful
                # message to the user.
                logger.warning("Connection problem! Please reconnect headset to phone!")
            except Exception:
                frame = None
                import traceback

                logger.error(traceback.format_exc())

            self._recent_frame = frame
            if frame:
                self._frame_size = (frame.width, frame.height)
                self.last_update = self.g_pool.get_timestamp()
                events["frame"] = frame
            elif (
                self.g_pool.get_timestamp() - self.last_update > self.ghost_mode_timeout
            ):
                logger.info("Device disconnected.")
                if self.online:
                    self.sensor.unlink()
                self.sensor = None
                self._source_id = None
                self._initial_refresh = True
                self.update_menu()
                self.last_update = self.g_pool.get_timestamp()
        else:
            time.sleep(self.get_frame_timeout / 1e3)

    # remote notifications
    def on_notification(self, sensor, event):
        # should only called if sensor was created
        if self._initial_refresh:
            self.sensor.set_control_value("streaming", True)
            self.sensor.refresh_controls()
            self._initial_refresh = False
        if event["subject"] == "error":
            # if not event['error_str'].startswith('err=-3'):
            logger.warning("Error {}".format(event["error_str"]))
            if "control_id" in event and event["control_id"] in self.sensor.controls:
                logger.debug(str(self.sensor.controls[event["control_id"]]))
        elif self.ui_initialized and (
            event["control_id"] not in self.control_id_ui_mapping
            or event["changes"].get("dtype") == "strmapping"
            or event["changes"].get("dtype") == "intmapping"
        ):
            self.update_menu()

    # local notifications
    def on_notify(self, notification):
        super().on_notify(notification)
        subject = notification["subject"]
        if subject.startswith("remote_recording.") and self.online:
            if "should_start" in subject and self.online:
                session_name = notification["session_name"]
                self.sensor.set_control_value("capture_session_name", session_name)
                self.sensor.set_control_value("local_capture", True)
            elif "should_stop" in subject:
                self.sensor.set_control_value("local_capture", False)

    @property
    def intrinsics(self):
        if self._intrinsics is None or self._intrinsics.resolution != self.frame_size:
            self._intrinsics = Camera_Model.from_file(
                self.g_pool.user_dir, self.name, self.frame_size
            )
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        self._intrinsics = model

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def frame_rate(self):
        if self.online:
            # FIXME: Hacky way to calculate frame rate. Depends on control option's caption
            fr_ctrl = self.sensor.controls.get("CAM_FRAME_RATE_CONTROL")
            if fr_ctrl:
                current_fr = fr_ctrl.get("value")
                map_ = {
                    mapping["value"]: mapping["caption"]
                    for mapping in fr_ctrl.get("map", [])
                }
                current_fr_cap = map_[current_fr].replace("Hz", "").strip()
                return float(current_fr_cap)

        return self._frame_rate

    @property
    def jpeg_support(self):
        return isinstance(self._recent_frame, ndsi.frame.JPEGFrame)

    def get_init_dict(self):
        settings = super().get_init_dict()
        settings["frame_rate"] = self.frame_rate
        settings["frame_size"] = self.frame_size
        if self.online:
            settings["sensor_name"] = self.sensor.name
            settings["host_name"] = self.sensor.host_name
        else:
            settings["sensor_name"] = self._sensor_name
            settings["host_name"] = self._host_name
        return settings

    def init_ui(self):
        super().init_ui()
        self.ui_initialized = True

    def deinit_ui(self):
        self.ui_initialized = False
        super().deinit_ui()

    def add_controls_to_menu(self, menu, controls):
        from pyglui import ui

        # closure factory
        def make_value_change_fn(ctrl_id):
            def initiate_value_change(val):
                logger.debug(f"{self.sensor}: {ctrl_id} >> {val}")
                self.sensor.set_control_value(ctrl_id, val)

            return initiate_value_change

        for ctrl_id, ctrl_dict in controls:
            try:
                dtype = ctrl_dict["dtype"]
                ctrl_ui = None
                if dtype == "string":
                    ctrl_ui = ui.Text_Input(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        setter=make_value_change_fn(ctrl_id),
                    )
                elif dtype == "integer" or dtype == "float":
                    convert_fn = int if dtype == "integer" else float
                    ctrl_ui = ui.Slider(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        min=convert_fn(ctrl_dict.get("min", 0)),
                        max=convert_fn(ctrl_dict.get("max", 100)),
                        step=convert_fn(ctrl_dict.get("res", 0.0)),
                        setter=make_value_change_fn(ctrl_id),
                    )
                elif dtype == "bool":
                    ctrl_ui = ui.Switch(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        on_val=ctrl_dict.get("max", True),
                        off_val=ctrl_dict.get("min", False),
                        setter=make_value_change_fn(ctrl_id),
                    )
                elif dtype == "strmapping" or dtype == "intmapping":
                    desc_list = ctrl_dict["map"]
                    labels = [desc["caption"] for desc in desc_list]
                    selection = [desc["value"] for desc in desc_list]
                    ctrl_ui = ui.Selector(
                        "value",
                        ctrl_dict,
                        label=ctrl_dict["caption"],
                        labels=labels,
                        selection=selection,
                        setter=make_value_change_fn(ctrl_id),
                    )
                if ctrl_ui:
                    ctrl_ui.read_only = ctrl_dict.get("readonly", False)
                    self.control_id_ui_mapping[ctrl_id] = ctrl_ui
                    menu.append(ctrl_ui)
                else:
                    logger.error(f"Did not generate UI for {ctrl_id}")
            except Exception:
                logger.error(f"Exception for control:\n{ctrl_dict}")
                import traceback as tb

                tb.print_exc()
        return menu

    def ui_elements(self):
        ui_elements = []
        ui_elements.append(
            ui.Info_Text(f"Camera: {self._sensor_name} @ {self._host_name}")
        )

        if not self.sensor:
            ui_elements.append(ui.Info_Text("Camera disconnected!"))
            return ui_elements

        uvc_controls = []
        other_controls = []
        for entry in iter(sorted(self.sensor.controls.items())):
            if entry[0].startswith("UVC"):
                uvc_controls.append(entry)
            else:
                other_controls.append(entry)

        uvc_menu = ui.Growing_Menu("UVC Controls")
        self.control_id_ui_mapping = {}
        if other_controls:
            self.add_controls_to_menu(ui_elements, other_controls)
        if uvc_controls:
            self.add_controls_to_menu(uvc_menu, uvc_controls)
        else:
            uvc_menu.append(ui.Info_Text("No UVC settings found."))
        ui_elements.append(uvc_menu)

        ui_elements.append(
            ui.Button("Reset to default values", self.sensor.reset_all_control_values)
        )

        return ui_elements

    def cleanup(self):
        if self.online:
            self.sensor.unlink()
        self.sensor = None


class NDSI_Manager(Base_Manager):
    """Enumerates and activates NDSI video sources"""

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.network = ndsi.Network(
            formats={ndsi.DataFormat.V3, ndsi.DataFormat.V4}, callbacks=(self.on_event,)
        )
        self.network.start()
        self._recover_in = 3
        self._rejoin_in = 400
        self.cam_selection_lut = {
            "eye0": ["ID0", "PI right"],
            "eye1": ["ID1", "PI left"],
            "world": ["ID2", "Logitech", "PI world"],
        }

    def cleanup(self):
        self.network.stop()

    def get_devices(self):
        # store hosts in dict to remove duplicates from multiple sensors
        active_hosts = {
            s["host_uuid"]: s["host_name"]
            for s in self.network.sensors.values()
            if s["sensor_type"] == "video"
        }
        return [
            SourceInfo(label=host_name, manager=self, key=f"host.{host_uuid}")
            for host_uuid, host_name in active_hosts.items()
        ]

    def get_cameras(self):
        return [
            SourceInfo(
                label=f"{s['sensor_name']} @ PM {s['host_name']}",
                manager=self,
                key=f"sensor.{s['sensor_uuid']}",
            )
            for s in self.network.sensors.values()
            if s["sensor_type"] == "video"
        ]

    def activate(self, key):
        source_type, uid = key.split(".", maxsplit=1)
        if source_type == "host":
            self.notify_all(
                {"subject": "backend.ndsi.auto_activate_source", "host_uid": uid}
            )
        elif source_type == "sensor":
            self.activate_source(source_uid=uid)

    def activate_source(self, source_uid):
        if not source_uid:
            return
        settings = {
            "frame_size": self.g_pool.capture.frame_size,
            "frame_rate": self.g_pool.capture.frame_rate,
            "source_id": source_uid,
        }
        if self.g_pool.process == "world":
            self.notify_all(
                {"subject": "start_plugin", "name": "NDSI_Source", "args": settings}
            )
        else:
            self.notify_all(
                {
                    "subject": "start_eye_plugin",
                    "target": self.g_pool.process,
                    "name": "NDSI_Source",
                    "args": settings,
                }
            )

    def auto_activate_source(self, host_uid):
        host_sensors = [
            sensor
            for sensor in self.network.sensors.values()
            if (sensor["sensor_type"] == "video" and sensor["host_uuid"] == host_uid)
        ]

        if not host_sensors:
            logger.warning("No devices available on the remote host.")
            return

        name_patterns = self.cam_selection_lut[self.g_pool.process]
        matching_cams = [
            sensor
            for sensor in host_sensors
            if any(pattern in sensor["sensor_name"] for pattern in name_patterns)
        ]

        if not matching_cams:
            logger.warning("The default device was not found on the remote host.")
            return

        cam = matching_cams[0]
        self.activate_source(cam["sensor_uuid"])

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def recent_events(self, events):
        self.poll_events()

        if (
            isinstance(self.g_pool.capture, NDSI_Source)
            and not self.g_pool.capture.sensor
        ):
            if self._recover_in <= 0:
                self.recover()
                self._recover_in = int(2 * 1e3 / self.g_pool.capture.get_frame_timeout)
            else:
                self._recover_in -= 1

            if self._rejoin_in <= 0:
                logger.debug("Rejoining network...")
                self.network.rejoin()
                # frame-timeout independent timer
                self._rejoin_in = int(10 * 1e3 / self.g_pool.capture.get_frame_timeout)
            else:
                self._rejoin_in -= 1

    def on_event(self, caller, event):
        if event["subject"] == "detach":
            logger.debug("detached: %s" % event)
            sensors = [s for s in self.network.sensors.values()]

        elif event["subject"] == "attach":
            if event["sensor_type"] == "video":
                logger.debug(f"attached: {event}")
                self.notify_all({"subject": "backend.ndsi_source_found"})

    def recover(self):
        if isinstance(self.g_pool.capture, NDSI_Source):
            self.g_pool.capture.recover(self.network)

    def on_notify(self, n):
        """Starts appropriate NDSI sources.

        Reacts to notification:
            ``backend.ndsi_source_found``: Check if recovery is possible
            ``backend.ndsi.auto_activate_source``: Auto activate best source for process

        Emmits notifications:
            ``backend.ndsi_source_found``: New NDSI source available
            ``backend.ndsi.auto_activate_source``: All NDSI managers should auto activate a source
            ``start_(eye_)plugin``: Starts NDSI sources
        """

        super().on_notify(n)

        if (
            n["subject"].startswith("backend.ndsi_source_found")
            and isinstance(self.g_pool.capture, NDSI_Source)
            and not self.g_pool.capture.sensor
        ):
            self.recover()

        if n["subject"] == "backend.ndsi.auto_activate_source":
            self.auto_activate_source(n["host_uid"])
