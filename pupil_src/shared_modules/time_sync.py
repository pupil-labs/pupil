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
import random
from heapq import heappush
from socket import gethostname
from urllib.parse import urlparse

import os_utils
from network_time_sync import Clock_Sync_Follower, Clock_Sync_Master
from plugin import Plugin
from pyglui import ui
from pyre import Pyre

os_utils.patch_pyre_zhelper_cdll()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from pyre import __version__

    assert __version__ >= "0.3.4"
except (ImportError, AssertionError):
    raise Exception("Pyre version is to old. Please upgrade")


__protocol_version__ = "v1"


class Clock_Service:
    """Represents a remote clock service and is sortable by rank."""

    def __init__(self, uuid, name, rank, port):
        super().__init__()
        self.uuid = uuid
        self.rank = rank
        self.port = port
        self.name = name

    def __repr__(self):
        return f"{self.rank:.2f}:{self.name}"

    def __lt__(self, other):
        # "smallest" object has highest rank
        return (self.rank > other.rank) if isinstance(other, Clock_Service) else False


class Time_Sync(Plugin):
    """Synchronize time across local network.

    Implements the Pupil Time Sync protocol.
    Acts as clock service and as follower if required.
    See `time_sync_spec.md` for details.
    """

    icon_chr = chr(0xEC15)
    icon_font = "pupil_icons"

    def __init__(
        self, g_pool, node_name=None, sync_group_prefix="default", base_bias=1.0
    ):
        super().__init__(g_pool)
        self.sync_group_prefix = sync_group_prefix
        self.discovery = None

        self.leaderboard = []
        self.has_been_master = 0.0
        self.has_been_synced = 0.0
        self.tie_breaker = random.random()
        self.base_bias = base_bias

        self.sync_group_members = {}

        self.master_service = Clock_Sync_Master(self.g_pool.get_timestamp)
        self.follower_service = None  # only set if there is a better server than us

        self.restart_discovery(node_name)

    @property
    def sync_group(self):
        return self.sync_group_prefix + "-time_sync-" + __protocol_version__

    @sync_group.setter
    def sync_group(self, full_name):
        self.sync_group_prefix = full_name.rsplit(
            "-time_sync-" + __protocol_version__, maxsplit=1
        )[0]

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Network Time Sync"
        help_str = "Synchonize time of Pupil Captures across the local network."
        self.menu.append(ui.Info_Text("Protocol version: " + __protocol_version__))

        self.menu.append(ui.Info_Text(help_str))
        help_str = "All pupil nodes of one group share a Master clock."
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(
            ui.Text_Input(
                "node_name", self, label="Node Name", setter=self.restart_discovery
            )
        )
        self.menu.append(
            ui.Text_Input(
                "sync_group_prefix",
                self,
                label="Sync Group",
                setter=self.change_sync_group,
            )
        )

        def sync_status():
            if self.follower_service:
                return str(self.follower_service)
            else:
                return "Clock Master"

        self.menu.append(
            ui.Text_Input(
                "sync status", getter=sync_status, setter=lambda _: _, label="Status"
            )
        )

        def set_bias(bias):
            if bias < 0:
                bias = 0.0
            self.base_bias = bias
            self.announce_clock_master_info()
            self.evaluate_leaderboard()

        help_str = "The clock service with the highest bias becomes clock master."
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(
            ui.Text_Input("base_bias", self, label="Master Bias", setter=set_bias)
        )
        self.menu.append(
            ui.Text_Input("leaderboard", self, label="Master Nodes in Group")
        )
        self.sync_group_members_menu = ui.Growing_Menu("Sync Group Members")
        self.menu.append(self.sync_group_members_menu)

    def recent_events(self, events):
        should_announce = False
        for evt in self.discovery.recent_events():
            if evt.type == "SHOUT":
                try:
                    self.update_leaderboard(
                        evt.peer_uuid, evt.peer_name, float(evt.msg[0]), int(evt.msg[1])
                    )
                except Exception as e:
                    logger.debug(f"Garbage raised `{e}` -- dropping.")
                self.evaluate_leaderboard()
            elif evt.type == "JOIN" and evt.group == self.sync_group:
                should_announce = True
                self.insert_sync_group_member(evt.peer_uuid, evt.peer_name)
            elif (
                evt.type == "LEAVE" and evt.group == self.sync_group
            ) or evt.type == "EXIT":
                self.remove_from_leaderboard(evt.peer_uuid)
                self.evaluate_leaderboard()
                self.remove_sync_group_member(evt.peer_uuid)

        if should_announce:
            self.announce_clock_master_info()

        if (
            not self.has_been_synced
            and self.follower_service
            and self.follower_service.in_sync
        ):
            self.has_been_synced = 1.0
            self.announce_clock_master_info()
            self.evaluate_leaderboard()

    def update_leaderboard(self, uuid, name, rank, port):
        for cs in self.leaderboard:
            if cs.uuid == uuid:
                if (cs.rank != rank) or (cs.port != port):
                    self.remove_from_leaderboard(cs.uuid)
                    break
                else:
                    # no changes. Just leave as is
                    return

        # clock service was not encountered before or has changed adding it to leaderboard
        cs = Clock_Service(uuid, name, rank, port)
        heappush(self.leaderboard, cs)
        logger.debug(f"{cs} added")

    def remove_from_leaderboard(self, uuid):
        for cs in self.leaderboard:
            if cs.uuid == uuid:
                self.leaderboard.remove(cs)
                logger.debug(f"{cs} removed")
                break

    def evaluate_leaderboard(self):
        if not self.leaderboard:
            logger.debug("nobody on the leader board.")
            return

        current_leader = self.leaderboard[0]
        if self.discovery.uuid() != current_leader.uuid:
            # we are not the leader!
            leader_ep = self.discovery.peer_address(current_leader.uuid)
            leader_addr = urlparse(leader_ep).netloc.split(":")[0]
            if self.follower_service is None:
                # make new follower
                self.follower_service = Clock_Sync_Follower(
                    leader_addr,
                    port=current_leader.port,
                    interval=10,
                    time_fn=self.get_time,
                    jump_fn=self.jump_time,
                    slew_fn=self.slew_time,
                )
            else:
                # update follower_service
                self.follower_service.host = leader_addr
                self.follower_service.port = current_leader.port
            return

        # we are the leader
        logger.debug("we are the leader")
        if self.follower_service is not None:
            self.follower_service.terminate()
            self.follower_service = None

        if not self.has_been_master:
            self.has_been_master = 1.0
            logger.debug(f"Become clock master with rank {self.rank}")
            self.announce_clock_master_info()

    def insert_sync_group_member(self, uuid, name):
        member_text = ui.Info_Text(name)
        self.sync_group_members[uuid] = member_text
        self.sync_group_members_menu.append(member_text)
        self.sync_group_members_menu.elements.sort(
            key=lambda text_field: text_field.text
        )

    def insert_all_sync_group_members_from_group(self, group):
        for uuid in self.discovery.peers_by_group(group):
            name = self.discovery.get_peer_name(uuid)
            self.insert_sync_group_member(uuid, name)

    def remove_all_sync_group_members(self):
        for uuid in list(self.sync_group_members.keys()):
            self.remove_sync_group_member(uuid)

    def remove_sync_group_member(self, uuid):
        try:
            self.sync_group_members_menu.remove(self.sync_group_members[uuid])
            del self.sync_group_members[uuid]
        except KeyError:
            logger.debug("Peer has already been removed from members list.")

    def announce_clock_master_info(self):
        self.discovery.shout(
            self.sync_group,
            [repr(self.rank).encode(), repr(self.master_service.port).encode()],
        )
        self.update_leaderboard(
            self.discovery.uuid(), self.node_name, self.rank, self.master_service.port
        )

    @property
    def rank(self):
        return (
            4 * self.base_bias
            + 2 * self.has_been_master
            + self.has_been_synced
            + self.tie_breaker
        )

    def get_time(self):
        return self.g_pool.get_timestamp()

    def slew_time(self, offset):
        self.g_pool.timebase.value += offset

    def jump_time(self, offset):
        ok_to_change = True
        for p in self.g_pool.plugins:
            if p.class_name == "Recorder":
                if p.running:
                    ok_to_change = False
                    logger.error(
                        "Request to change timebase during recording ignored. Turn off recording first."
                    )
                    break
        if ok_to_change:
            self.slew_time(offset)
            logger.info(f"Pupil Sync has adjusted the clock by {offset}s")
            return True
        else:
            return False

    def restart_discovery(self, name):
        if self.discovery:
            if self.discovery.name() == name:
                return
            else:
                self.remove_all_sync_group_members()
                self.discovery.leave(self.sync_group)
                self.discovery.stop()
                self.leaderboard = []

        self.node_name = name or gethostname()
        self.discovery = Pyre(self.node_name)
        # Either joining network for the first time or rejoining the same group.
        self.discovery.join(self.sync_group)
        self.discovery.start()
        self.announce_clock_master_info()

    def change_sync_group(self, new_group_prefix):
        if new_group_prefix != self.sync_group_prefix:
            self.remove_all_sync_group_members()
            self.discovery.leave(self.sync_group)
            self.leaderboard = []
            if self.follower_service:
                self.follower_service.terminate()
                self.follower = None
            self.sync_group_prefix = new_group_prefix
            self.discovery.join(self.sync_group)
            self.insert_all_sync_group_members_from_group(self.sync_group)
            self.announce_clock_master_info()

    def deinit_ui(self):
        for uuid in list(self.sync_group_members.keys()):
            self.remove_sync_group_member(uuid)
        self.remove_menu()

    def get_init_dict(self):
        return {
            "node_name": self.node_name,
            "sync_group_prefix": self.sync_group_prefix,
            "base_bias": self.base_bias,
        }

    def cleanup(self):
        self.discovery.leave(self.sync_group)
        self.discovery.stop()
        self.master_service.stop()
        if self.follower_service:
            self.follower_service.stop()
        self.master_service = None
        self.follower_service = None
        self.discovery = None
