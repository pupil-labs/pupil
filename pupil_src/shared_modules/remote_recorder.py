"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

import ndsi
from pyglui import ui

from plugin import Plugin

logger = logging.getLogger(__name__)


class Remote_Recording_State:
    __slots__ = ["sensor"]
    _UNDEFINED = "Undefined"

    def __init__(self, network, attach_event):
        self.sensor = network.sensor(
            attach_event["sensor_uuid"], callbacks=(self.check_error,)
        )

    def detach(self):
        self.sensor.unlink()
        del self.sensor

    @property
    def label(self):
        return self.sensor.host_name

    @property
    def is_recording(self):
        try:
            return self.sensor.controls["local_capture"]["value"]
        except KeyError:
            return False

    @is_recording.setter
    def is_recording(self, should_be_recording):
        self.sensor.set_control_value("local_capture", should_be_recording)

    @property
    def session_name(self):
        try:
            return self.sensor.controls["capture_session_name"]["value"]
        except KeyError:
            return "Unknown"

    @session_name.setter
    def session_name(self, session_name):
        self.sensor.set_control_value("capture_session_name", session_name)

    @property
    def remote_version(self):
        try:
            return self.sensor.controls["version"]["value"]
        except KeyError:
            return self._UNDEFINED

    @property
    def supports_remote_control(self) -> bool:
        is_v4_format = self.sensor.format is ndsi.DataFormat.V4
        is_remote_version_undefined = self.remote_version == self._UNDEFINED
        out_of_date = is_v4_format and is_remote_version_undefined
        return not out_of_date

    def poll_updates(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()

    def check_error(self, sensor, notification):
        if notification["subject"] == "error":
            logger.error(notification["error_str"].split("\n")[0])
            logger.debug(notification["error_str"])


class Remote_Recorder_Core:
    __slots__ = ["_attached_rec_states", "_network", "num_states_changed"]

    def __init__(self, num_states_changed_callback):
        assert callable(num_states_changed_callback)
        self.num_states_changed = num_states_changed_callback

        self._attached_rec_states = {}
        self._network = ndsi.Network(
            formats={ndsi.DataFormat.V3, ndsi.DataFormat.V4}, callbacks=(self.on_event,)
        )
        self._network.start()

    def poll_network_events(self):
        while self._network.has_events:
            self._network.handle_event()

        for rec_state in self._attached_rec_states.values():
            rec_state.poll_updates()

    def on_event(self, caller, event):
        if event["subject"] == "attach" and event["sensor_type"] == "hardware":
            self.attach_rec_state(event)
        elif (
            event["subject"] == "detach"
            and event["sensor_uuid"] in self._attached_rec_states
        ):
            self.detach_rec_state(event)
        else:
            return
        self.num_states_changed()

    def attach_rec_state(self, event):
        rec_state = Remote_Recording_State(self._network, event)
        self._attached_rec_states[event["sensor_uuid"]] = rec_state

    def detach_rec_state(self, event):
        self._attached_rec_states[event["sensor_uuid"]].detach()
        del self._attached_rec_states[event["sensor_uuid"]]

    def broadcast_preferred_session_name(self, preferred_session_name):
        for rec_state in self._attached_rec_states.values():
            if rec_state.sensor.format is ndsi.DataFormat.V3:
                rec_state.session_name = preferred_session_name

    def cleanup(self):
        for state in self._attached_rec_states.values():
            state.detach()
        self._attached_rec_states.clear()
        self._network.stop()

    def rec_states_sorted(self, key=lambda state: state.label):
        return sorted(self._attached_rec_states.values(), key=key)


class Remote_Recorder(Plugin):

    order = 0.3
    uniqueness = "by_class"
    icon_chr = chr(0xEC16)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, preferred_session_name="local_recording"):
        super().__init__(g_pool)
        self._core = Remote_Recorder_Core(num_states_changed_callback=self.refresh_menu)
        self.preferred_session_name = preferred_session_name
        self._switch_elements = {}

    def recent_events(self, events):
        self._core.poll_network_events()
        for rec_state, switch in self._switch_elements.items():
            # NOTE: triggers ui refresh if value changes
            switch.read_only = not rec_state.supports_remote_control

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Remote Recorder"
        self.refresh_menu()

    def refresh_menu(self):
        del self.menu[:]
        self._switch_elements.clear()
        self.menu.append(
            ui.Info_Text(
                "Start and stop recording sessions remotely"
                " on available Pupil Mobile hosts."
            )
        )

        self.append_preferred_session_name_setter()
        self.menu.append(ui.Separator())

        for rec_state in self._core.rec_states_sorted():
            self.append_rec_state_switch(rec_state)
            if rec_state.sensor.format is ndsi.DataFormat.V3:
                self.append_session_name_view(rec_state)

        self.menu.append(
            ui.Info_Text(
                "Greyed out devices are out-of-date and do not support starting/"
                "stopping recordings remotely. Update the Pupil Invisible Companion "
                "app to enable its support."
            )
        )

    def append_preferred_session_name_setter(self):
        self.menu.append(
            ui.Text_Input(
                "preferred_session_name", self, label="Preferred session name"
            )
        )
        self.menu.append(
            ui.Button(
                "Broadcast preferred session name",
                self.broadcast_preferred_session_name,
            )
        )

    def append_rec_state_switch(self, rec_state):
        label = rec_state.label
        view = ui.Switch("is_recording", rec_state, label=label)
        view.read_only = not rec_state.supports_remote_control
        self.menu.append(view)
        self._switch_elements[rec_state] = view

    def append_session_name_view(self, rec_state):
        view = ui.Text_Input("session_name", rec_state, label="Session name")
        view.read_only = True
        self.menu.append(view)

    def broadcast_preferred_session_name(self):
        self._core.broadcast_preferred_session_name(self.preferred_session_name)

    def get_init_dict(self):
        return {"preferred_session_name": self.preferred_session_name}

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self._core.cleanup()
