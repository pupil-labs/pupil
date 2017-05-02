'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


from plugin import Plugin
from pyglui import ui
from socket import gethostname
from heapq import heappush
from pyre import Pyre
from urllib.parse import urlparse
from network_time_sync import Clock_Sync_Master, Clock_Sync_Follower
import random

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Clock_Service(object):
    """Represents a remote clock service and is sortable by rank."""
    def __init__(self, uuid, rank, port):
        super(Clock_Service, self).__init__()
        self.uuid = uuid
        self.rank = rank
        self.port = port

    def __str__(self):
        return '<Clock Service: {} rank={} port={}>'.format(self.uuid.hex, self.rank, self.port)

    def __lt__(self, other):
        # "smallest" object has highest rank
        return (self.rank > other.rank) if isinstance(other, Clock_Service) else False


class Time_Sync(Plugin):
    """Synchronize time across local network.

    Implements the Pupil Time Sync protocol.
    Acts as clock service and as follower if required.
    See `time_sync_spec.md` for details.
    """

    def __init__(self, g_pool, node_name=None, sync_group='time_sync_default', base_bias=1.):
        super().__init__(g_pool)
        self.menu = None
        self.sync_group = sync_group
        self.discovery = None

        self.leaderboard = []
        self.has_been_master = 0.
        self.has_been_synced = 0.
        self.tie_breaker = random.random()
        self.base_bias = base_bias

        self.own_service = Clock_Sync_Master(self.g_pool.get_timestamp)
        self.followed_service = None  # only set if there is a better server than us

        self.restart_discovery(node_name)

    def init_gui(self):
        def close():
            self.alive = False

        help_str = "Synchonize time of Pupil Captures across the local network."
        self.menu = ui.Growing_Menu('Network Time Sync')
        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text(help_str))
        help_str = "All pupil nodes of one group share a Master clock."
        self.menu.append(ui.Info_Text(help_str))
        # self.menu.append(ui.Text_Input('sync status',getter=sync_status_info,setter=lambda _: _))
        # self.menu[-1].read_only = True
        self.menu.append(ui.Text_Input('node_name', self, label='Node Name', setter=self.restart_discovery))
        self.menu.append(ui.Text_Input('sync_group', self, label='Sync Group', setter=self.change_sync_group))

        def sync_status():
            if self.followed_service:
                status = 'In Sync' if self.followed_service.in_sync else 'Syncing'
                return 'Clock Follower — ' + status
            else:
                return 'Clock Master'
        self.menu.append(ui.Text_Input('sync status', getter=sync_status, setter=lambda _: _, label='Status'))

        def set_bias(bias):
            if bias < 0:
                bias = 0.
            if bias != self.base_bias:
                self.base_bias = bias
                self.make_clock_service_announcement()
                self.evaluate_leaderboard()

        help_str = "The clock service with the highest bias becomes clock master. The base bias influences this value greatly."
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('base_bias', self, label='Base Bias', setter=set_bias))
        self.g_pool.sidebar.append(self.menu)

    def recent_events(self, events):
        should_announce = False
        for evt in self.discovery.recent_events():
            if evt.type == 'SHOUT':
                try:
                    self.update_clock_service(evt.peer_uuid, float(evt.msg[0]), int(evt.msg[1]))
                except Exception as e:
                    logger.debug('Garbage raised `{}` -- dropping.'.format(e))
            elif evt.type == 'JOIN' and evt.group == self.sync_group:
                should_announce = True
            elif (evt.type == 'LEAVE' and evt.group == self.sync_group) or evt.type == 'EXIT':
                self.remove_clock_service(evt.peer_uuid)

        if not self.has_been_synced and self.followed_service and self.followed_service.in_sync:
            self.has_been_synced = 1.
            self.evaluate_leaderboard()
            should_announce = True

        if should_announce:
            self.make_clock_service_announcement()

    def update_clock_service(self, uuid, rank, port):
        for cs in self.leaderboard:
            if cs.uuid == uuid:
                if (cs.rank != rank) or (cs.port != port):
                    cs.rank = rank
                    cs.port = port
                    self.evaluate_leaderboard()
                return

        # clock service was not encountered before -- adding it to leaderboard
        cs = Clock_Service(uuid, rank, port)
        heappush(self.leaderboard, cs)
        logger.debug('{} added'.format(cs))
        self.evaluate_leaderboard()

    def remove_clock_service(self, uuid):
        for cs in self.leaderboard:
            if cs.uuid == uuid:
                self.leaderboard.remove(cs)
                logger.debug('{} removed'.format(cs))
                self.evaluate_leaderboard()
                return  # stop iteration

    def evaluate_leaderboard(self):
        if self.leaderboard:
            current_leader = self.leaderboard[0]
            if self.own_rank < current_leader.rank:
                leader_ep = self.discovery.peer_address(current_leader.uuid)
                leader_addr = urlparse(leader_ep).netloc.split(':')[0]
                if self.followed_service is None:
                    self.followed_service = Clock_Sync_Follower(leader_addr,
                                                                port=current_leader.port,
                                                                interval=10,
                                                                time_fn=self.get_time,
                                                                jump_fn=self.jump_time,
                                                                slew_fn=self.slew_time)
                    logger.debug('Become clock follower with rank {}'.format(self.own_rank))
                else:
                    self.followed_service.host = leader_addr
                    self.followed_service.port = current_leader.port
                return
        # self should be clockmaster
        if self.followed_service is not None:
            self.followed_service.terminate()
            self.followed_service = None

        if not self.has_been_master:
            self.has_been_master = 1.
            self.evaluate_leaderboard()
        else:
            logger.debug('Become clock master with rank {}'.format(self.own_rank))
            self.make_clock_service_announcement()

    def make_clock_service_announcement(self):
        self.discovery.shout(self.sync_group, [repr(self.own_rank).encode(),
                                               repr(self.own_service.port).encode()])

    @property
    def own_rank(self):
        return 4*self.base_bias + 2*self.has_been_master + self.has_been_synced + self.tie_breaker

    def get_time(self):
        return self.g_pool.get_timestamp()

    def slew_time(self, offset):
        self.g_pool.timebase.value += offset

    def jump_time(self, offset):
        ok_to_change = True
        for p in self.g_pool.plugins:
            if p.class_name == 'Recorder':
                if p.running:
                    ok_to_change = False
                    logger.error("Request to change timebase during recording ignored. Turn off recording first.")
                    break
        if ok_to_change:
            self.slew_time(offset)
            logger.info("Pupil Sync has adjusted the clock by {}s".format(offset))
            return True
        else:
            return False

    def restart_discovery(self, name):
        if self.discovery:
            if self.discovery.name() == name:
                return
            self.discovery.leave(self.sync_group)
            self.discovery.stop()

        self.node_name = name or gethostname()
        self.discovery = Pyre(self.node_name)
        self.discovery.join(self.sync_group)
        self.discovery.start()

    def change_sync_group(self, new_group):
        if new_group != self.sync_group:
            self.discovery.leave(self.sync_group)
            self.leaderboard = []
            self.sync_group = new_group
            self.discovery.join(new_group)
            if not self.discovery.peers_by_group(new_group):
                # new_group is empty
                self.evaluate_leaderboard()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {'node_name': self.node_name,
                'sync_group': self.sync_group,
                'base_bias': self.base_bias}

    def cleanup(self):
        self.deinit_gui()
        self.discovery.leave(self.sync_group)
        self.discovery.stop()
        self.own_service.stop()
