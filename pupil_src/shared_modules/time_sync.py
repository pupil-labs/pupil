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
import logging
logger = logging.getLogger(__name__)

from network_time_sync import Clock_Sync_Master,Clock_Sync_Follower
import random

class Time_Sync(Plugin):
    """Synchronize time of Actors
        across local network.
    """


    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.menu = None

        #variables for the time sync logic
        self.time_sync_node = None

        #constants for the time sync logic
        self._ti_break = random.random()/10.
        self.master_announce_interval = 5
        self.master_announce_timeout = self.master_announce_interval * 4
        self.master_announce_timeout_notification = {'subject':"time_sync.master_announce_timeout", 'delay':self.master_announce_timeout}
        self.master_announce_interval_notification = {'subject':"time_sync.master_announce_interval", 'delay':self.master_announce_interval}

        self.notify_all(self.master_announce_timeout_notification)

    @property
    def is_master(self):
        return isinstance(self.time_sync_node,Clock_Sync_Master)

    @property
    def is_follower(self):
        return isinstance(self.time_sync_node,Clock_Sync_Follower)

    @property
    def is_nothing(self):
        return self.time_sync_node is None

    def clock_master_worthiness(self):
        '''
        How worthy am I to be the clock master?
        A measure 0 (unworthy) to 1 (destined)

        range is from 0. - 0.9 the rest is reserved for ti-breaking
        '''

        worthiness = 0.
        if self.g_pool.timebase.value != 0:
            worthiness += 0.4
        worthiness +=self._ti_break
        return worthiness

    ###time sync fns these are used by the time sync node to get and adjust time
    def get_unadjusted_time(self):
        #return time not influced by outside clocks.
        return self.g_pool.get_now()

    def get_time(self):
        return self.g_pool.get_timestamp()

    def slew_time(self,offset):
        self.g_pool.timebase.value +=offset

    def jump_time(self,offset):
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


    def init_gui(self):

        def close():
            self.alive = False

        def sync_status_info():
            if self.time_sync_node is None:
                return 'Waiting for time sync msg.'
            else:
                return str(self.time_sync_node)

        help_str = "Synchonize time of Pupil captures across the local network."
        self.menu = ui.Growing_Menu('Network Time Sync')
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Info_Text(help_str))
        help_str = "All pupil nodes of one group share a Master clock."
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('sync status',getter=sync_status_info,setter=lambda _: _))
        # self.menu[-1].read_only = True
        self.g_pool.sidebar.append(self.menu)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None


    def on_notify(self,notification):
        """Synchronize time of Actors across local network.

        The notification scheme is used to handle interal timing
        and to talk to remote pers via the `Pupil_Groups` plugin.

        Reacts to notifications:
            ``time_sync.master_announcement``: React accordingly to annouce notification from remote peer.
            ``time_sync.master_announce_interval``: Re-annouce clock masterhood.
            ``time_sync.master_announce_timeout``: React accordingly when no master announcement has appeard whithin timeout.


        Emits notifications:
            ``time_sync.master_announcement``: Announce masterhood to remote peers (remote notification).
            ``time_sync.master_announce_interval``: Re-announce masterhood reminder (delayed notification).
            ``time_sync.master_announce_timeout``:  Timeout for foreind master announcement (delayed notification).

        """
        if notification['subject'].startswith('time_sync.master_announcement'):
            if self.is_master:
                if notification['worthiness'] > self.clock_master_worthiness():
                    #We need to yield.
                    self.time_sync_node.stop()
                    self.time_sync_node = None
                else:
                    #Denounce the lesser competition.
                    n = {   'subject':'time_sync.master_announcement',
                            'host':self.time_sync_node.host,
                            'port':self.time_sync_node.port,
                            'worthiness':self.clock_master_worthiness(),
                            'remote_notify':'all'
                        }
                    self.notify_all(n)

            if self.is_follower:
                # update follower info
                self.time_sync_node.host = notification['host']
                self.time_sync_node.port = notification['port']

            if self.is_nothing:
                # Create follower.
                logger.debug("Clock will sync with {}".format(notification['host']))
                self.time_sync_node = Clock_Sync_Follower(notification['host'],port=notification['port'],interval=10,time_fn=self.get_time,jump_fn=self.jump_time,slew_fn=self.slew_time)

            if not self.is_master:
                #(Re)set the timer.
                self.notify_all(self.master_announce_timeout_notification)


        elif notification['subject'].startswith('time_sync.master_announce_timeout'):
            if self.is_master:
                pass
            else:
                #We have not heard from a master in too long.
                logger.info("Elevate self to clock master.")
                self.time_sync_node = Clock_Sync_Master(self.g_pool.get_timestamp)
                n = {   'subject':'time_sync.master_announcement',
                        'host':self.time_sync_node.host,
                        'port':self.time_sync_node.port,
                        'worthiness':self.clock_master_worthiness(),
                        'remote_notify':'all'
                    }
                self.notify_all(n)
                self.notify_all(self.master_announce_interval_notification)


        elif notification['subject'].startswith('time_sync.master_announce_interval'):
            # The time has come to remind others of our master hood.
            if self.is_master:
                n = {   'subject':'time_sync.master_announcement',
                        'host':self.time_sync_node.host,
                        'port':self.time_sync_node.port,
                        'worthiness':self.clock_master_worthiness(),
                        'remote_notify':'all' }
                self.notify_all(n)
                # Set the next annouce timer.
                self.notify_all(self.master_announce_interval_notification)


    def get_init_dict(self):
        return {}

    def cleanup(self):
        if self.time_sync_node:
            self.time_sync_node.terminate()
        self.deinit_gui()


