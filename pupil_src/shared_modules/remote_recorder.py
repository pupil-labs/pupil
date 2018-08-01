"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
from time import localtime, strftime
from typing import Callable, Dict

import ndsi
from pyglui import ui

from plugin import Plugin

logger = logging.getLogger(__name__)


class Remote_Recording_State:
    __slots__ = ["sensor"]
    sensor: ndsi.Sensor

    def __init__(self, network, attach_event):
        self.sensor = network.sensor(attach_event["sensor_uuid"])

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

    def poll_updates(self):
        while self.sensor.has_notifications:
            self.sensor.handle_notification()


class Remote_Recorder_Core:
    __slots__ = ["_attached_rec_states", "_network", "num_states_changed"]
    _attached_rec_states: Dict[str, Remote_Recording_State]
    _network: ndsi.Network
    num_states_changed: Callable[[], None]

    def __init__(self, num_states_changed_callback):
        assert callable(num_states_changed_callback)
        self.num_states_changed = num_states_changed_callback

        self._attached_rec_states = {}
        self._network = ndsi.Network(callbacks=(self.on_event,))
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

    def cleanup(self):
        for state in self._attached_rec_states.values():
            state.detach()
        self._attached_rec_states.clear()
        self._network.stop()

    def rec_states_sorted(self, key=lambda state: state.label):
        return sorted(self._attached_rec_states.values(), key=key)


class Remote_Recorder(Plugin):

    order = .3
    uniqueness = "by_class"
    icon_chr = chr(0xec16)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._core = Remote_Recorder_Core(num_states_changed_callback=self.refresh_menu)

    def recent_events(self, events):
        self._core.poll_network_events()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Remote Recorder"
        self.refresh_menu()

    def refresh_menu(self):
        del self.menu[:]
        self.menu.append(
            ui.Info_Text(
                "Start and stop recording sessions remotely"
                " on available Pupil Mobile hosts."
            )
        )
        for rec_state in self._core.rec_states_sorted():
            self.add_state_ui_switch(rec_state)

    def add_state_ui_switch(self, rec_state):
        self.menu.append(ui.Switch("is_recording", rec_state, label=rec_state.label))

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self._core.cleanup()
