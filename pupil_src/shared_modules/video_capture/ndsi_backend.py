"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import time
import logging
from packaging.version import Version

import ndsi

from .base_backend import Base_Source, Base_Manager
from camera_models import load_intrinsics

try:
    from ndsi import __version__

    assert Version(__version__) >= Version("1.0.dev0")
    from ndsi import __protocol_version__
except (ImportError, AssertionError):
    raise Exception("pyndsi version is to old. Please upgrade") from None
logger = logging.getLogger(__name__)


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
        network=None,
        source_id=None,
        host_name=None,
        sensor_name=None,
    ):
        super().__init__(g_pool)
        self.sensor = None
        self._source_id = source_id
        self._sensor_name = sensor_name
        self._host_name = host_name
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self.has_ui = False
        self.control_id_ui_mapping = {}
        self.get_frame_timeout = 100  # ms
        self.ghost_mode_timeout = 10  # sec
        self._initial_refresh = True
        self.last_update = self.g_pool.get_timestamp()

        if not network:
            logger.debug(
                "No network reference provided. Capture is started "
                + "in ghost mode. No images will be supplied."
            )
            return

        self.recover(network)

        if not self.sensor or not self.sensor.supports_data_subscription:
            logger.error(
                "Init failed. Capture is started in ghost mode. "
                + "No images will be supplied."
            )
            self.cleanup()

        logger.debug("NDSI Source Sensor: %s" % self.sensor)

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
        return "{}".format(self._sensor_name)

    @property
    def host(self):
        return "{}".format(self._host_name)

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
                logger.info("Entering ghost mode")
                if self.online:
                    self.sensor.unlink()
                self.sensor = None
                self._source_id = None
                self._initial_refresh = True
                self.update_control_menu()
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
        elif self.has_ui and (
            event["control_id"] not in self.control_id_ui_mapping
            or event["changes"].get("dtype") == "strmapping"
            or event["changes"].get("dtype") == "intmapping"
        ):
            self.update_control_menu()

    # local notifications
    def on_notify(self, notification):
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
            self._intrinsics = load_intrinsics(
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
        self.add_menu()
        self.menu.label = "NDSI Source: {} @ {}".format(
            self._sensor_name, self._host_name
        )

        from pyglui import ui

        self.has_ui = True
        self.uvc_menu = ui.Growing_Menu("UVC Controls")
        self.update_control_menu()

    def deinit_ui(self):
        self.uvc_menu = None
        self.remove_menu()
        self.has_ui = False

    def add_controls_to_menu(self, menu, controls):
        from pyglui import ui

        # closure factory
        def make_value_change_fn(ctrl_id):
            def initiate_value_change(val):
                logger.debug("{}: {} >> {}".format(self.sensor, ctrl_id, val))
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
                    logger.error("Did not generate UI for {}".format(ctrl_id))
            except:
                logger.error("Exception for control:\n{}".format(ctrl_dict))
                import traceback as tb

                tb.print_exc()
        if len(menu) == 0:
            menu.append(ui.Info_Text("No {} settings found".format(menu.label)))
        return menu

    def update_control_menu(self):
        if not self.has_ui:
            return
        from pyglui import ui

        del self.menu[:]
        del self.uvc_menu[:]
        self.control_id_ui_mapping = {}
        if not self.sensor:
            self.menu.append(
                ui.Info_Text(
                    ("Sensor %s @ %s not available. " + "Running in ghost mode.")
                    % (self._sensor_name, self._host_name)
                )
            )
            return

        uvc_controls = []
        other_controls = []
        for entry in iter(sorted(self.sensor.controls.items())):
            if entry[0].startswith("UVC"):
                uvc_controls.append(entry)
            else:
                other_controls.append(entry)

        self.add_controls_to_menu(self.menu, other_controls)
        self.add_controls_to_menu(self.uvc_menu, uvc_controls)
        self.menu.append(self.uvc_menu)

        self.menu.append(
            ui.Button("Reset to default values", self.sensor.reset_all_control_values)
        )

    def cleanup(self):
        if self.online:
            self.sensor.unlink()
        self.sensor = None


class NDSI_Manager(Base_Manager):
    """Enumerates and activates Pupil Mobile video sources

    Attributes:
        network (ndsi.Network): NDSI Network backend
        selected_host (unicode): Selected host uuid
    """

    gui_name = "Pupil Mobile"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.network = ndsi.Network(
            formats={ndsi.DataFormat.V3}, callbacks=(self.on_event,)
        )
        self.network.start()
        self.selected_host = None
        self._recover_in = 3
        self._rejoin_in = 400
        self.should_select_host = None
        self.cam_selection_lut = {
            "eye0": ["ID0", "PI right"],
            "eye1": ["ID1", "PI left"],
            "world": ["ID2", "Logitech", "PI world"],
        }
        logger.warning("Make sure the `time_sync` plugin is loaded!")

    def cleanup(self):
        self.network.stop()

    def init_ui(self):
        self.add_menu()
        self.re_build_ndsi_menu()

    def deinit_ui(self):
        self.remove_menu()

    def view_host(self, host_uuid):
        if self.selected_host != host_uuid:
            self.selected_host = host_uuid
            self.re_build_ndsi_menu()

    def host_selection_list(self):
        devices = {
            s["host_uuid"]: s["host_name"]  # removes duplicates
            for s in self.network.sensors.values()
        }

        if devices:
            return list(devices.keys()), list(devices.values())
        else:
            return [None], ["No hosts found"]

    def source_selection_list(self):
        default = (None, "Select to activate")
        sources = [default] + [
            (s["sensor_uuid"], s["sensor_name"])
            for s in self.network.sensors.values()
            if (s["sensor_type"] == "video" and s["host_uuid"] == self.selected_host)
        ]
        return zip(*sources)

    def re_build_ndsi_menu(self):
        del self.menu[1:]
        from pyglui import ui

        ui_elements = []
        ui_elements.append(ui.Info_Text("Remote Pupil Mobile sources"))
        ui_elements.append(
            ui.Info_Text("Pupil Mobile Commspec v{}".format(__protocol_version__))
        )

        host_sel, host_sel_labels = self.host_selection_list()
        ui_elements.append(
            ui.Selector(
                "selected_host",
                self,
                selection=host_sel,
                labels=host_sel_labels,
                setter=self.view_host,
                label="Remote host",
            )
        )

        self.menu.extend(ui_elements)
        self.add_auto_select_button()

        if not self.selected_host:
            return
        ui_elements = []

        host_menu = ui.Growing_Menu("Remote Host Information")
        ui_elements.append(host_menu)

        src_sel, src_sel_labels = self.source_selection_list()
        host_menu.append(
            ui.Selector(
                "selected_source",
                selection=src_sel,
                labels=src_sel_labels,
                getter=lambda: None,
                setter=self.activate,
                label="Source",
            )
        )

        self.menu.extend(ui_elements)

    def activate(self, source_uid):
        if not source_uid:
            return
        settings = {
            "frame_size": self.g_pool.capture.frame_size,
            "frame_rate": self.g_pool.capture.frame_rate,
            "source_id": source_uid,
        }
        self.activate_source(settings)

    def auto_select_manager(self):
        super().auto_select_manager()
        self.notify_all(
            {
                "subject": "backend.ndsi_do_select_host",
                "target_host": self.selected_host,
                "delay": 0.4,
            }
        )

    def auto_activate_source(self):
        if not self.selected_host:
            return

        src_sel, src_sel_labels = self.source_selection_list()
        if len(src_sel) < 2:  # "Select to Activate" is always presenet as first element
            logger.warning("No device is available on the remote host.")
            return

        cam_ids = self.cam_selection_lut[self.g_pool.process]

        for cam_id in cam_ids:
            try:
                source_id = next(
                    src_sel[i] for i, lab in enumerate(src_sel_labels) if cam_id in lab
                )
                self.activate(source_id)
                break
            except StopIteration:
                source_id = None
        else:
            logger.warning("The default device was not found on the remote host.")

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
                self._rejoin_in = int(10 * 1e3 / self.g_pool.capture.get_frame_timeout)
            else:
                self._rejoin_in -= 1

    def on_event(self, caller, event):
        if event["subject"] == "detach":
            logger.debug("detached: %s" % event)
            sensors = [s for s in self.network.sensors.values()]
            if self.selected_host == event["host_uuid"]:
                if sensors:
                    self.selected_host = sensors[0]["host_uuid"]
                else:
                    self.selected_host = None
                self.re_build_ndsi_menu()

        elif event["subject"] == "attach":
            if event["sensor_type"] == "video":
                logger.debug("attached: {}".format(event))
                self.notify_all({"subject": "backend.ndsi_source_found"})

            if not self.selected_host and not self.should_select_host:
                self.selected_host = event["host_uuid"]
            elif self.should_select_host and event["sensor_type"] == "video":
                self.select_host(self.should_select_host)

            self.re_build_ndsi_menu()

    def activate_source(self, settings={}):
        settings["network"] = self.network
        self.g_pool.plugins.add(NDSI_Source, args=settings)

    def recover(self):
        self.g_pool.capture.recover(self.network)

    def on_notify(self, n):
        """Provides UI for the capture selection

        Reacts to notification:
            ``backend.ndsi_source_found``: Check if recovery is possible
            ``backend.ndsi_do_select_host``: Switches to selected host from other process

        Emmits notifications:
            ``backend.ndsi_source_found``
            ``backend.ndsi_do_select_host`
        """

        super().on_notify(n)

        if (
            n["subject"].startswith("backend.ndsi_source_found")
            and isinstance(self.g_pool.capture, NDSI_Source)
            and not self.g_pool.capture.sensor
        ):
            self.recover()

        if n["subject"].startswith("backend.ndsi_do_select_host"):
            self.select_host(n["target_host"])

    def select_host(self, selected_host):
        host_sel, _ = self.host_selection_list()
        if selected_host in host_sel:
            self.view_host(selected_host)
            self.should_select_host = None
            self.re_build_ndsi_menu()
            src_sel, _ = self.source_selection_list()
            # "Select to Activate" is always presenet as first element
            if len(src_sel) >= 2:
                self.auto_activate_source()

        else:
            self.should_select_host = selected_host

