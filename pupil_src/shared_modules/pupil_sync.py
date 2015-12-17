'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0) License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from plugin import Plugin
from pyglui import ui
import zmq
from pyre import Pyre, zhelper
from uuid import UUID
from time import sleep
from threading import Timer
import logging
logger = logging.getLogger(__name__)

from network_time_sync import Clock_Sync_Master,Clock_Sync_Follower

start_rec = "START_REC:"
stop_rec = "STOP_REC:"
start_cal = "START_CAL"
stop_cal = "STOP_CAL"
sync_time_master_announce = "SYNC_TIME_MASTER:"
user_event = "USREVENT:"
msg_delimeter  = '::'


# Pipe signals:
exit_thread = "EXIT_THREAD".encode('utf_8')

class Pupil_Sync(Plugin):
    """Synchonize behaviour of Pupil captures
        across the local network
    """
    def __init__(self, g_pool,name='unnamed Pupil',group='default group'):
        super(Pupil_Sync, self).__init__(g_pool)
        self.order = .01 #excecute first
        self.name = name
        self.group = group
        self.group_members = {}
        self.menu = None
        self.group_menu = None


        #variables for the time sync logic
        self.sync_node = None
        self.last_master_announce = self.get_monotonic_time()

        #constants for the time sync logic
        self.time_sync_announce_interval = 5
        self.time_sync_wait_interval_short = self.time_sync_announce_interval * 3
        self.time_sync_wait_interval_long = self.time_sync_announce_interval * 4


        self.context = zmq.Context()
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)




    def clock_master_worthiness(self):
        '''
        How worthy am I to be the clock master?
        A measure 0 (unworthy) to 1 (destined)
        tie-breaking is does via bigger uuid.int

        At current this is done via
        '''
        if self.g_pool.timebase.value == 0:
            return 0.
        else:
            return 0.5

    ###time sync fns these are used by the time sync node to get and adjust time
    def get_monotonic_time(self):
        #return time not influced by outside clocks.
        return self.g_pool.capture.get_now()

    def get_time(self):
        return self.g_pool.capture.get_timestamp()

    def set_time(self,offset):
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
            self.set_time(offset)
            logger.info("Pupil Sync has adjusted the clock by %ss"%offset)
            return True
        else:
            return False

    def sync_status_info(self):
        if self.sync_node is None:
            return 'Waiting for time sync msg.'
        else:
            return str(self.sync_node)

    def init_gui(self):
        help_str = "Synchonize behaviour of Pupil captures across the local network."
        self.menu = ui.Growing_Menu('Pupil Sync')
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('name',self,setter=self.set_name,label='Name'))
        self.menu.append(ui.Text_Input('group',self,setter=self.set_group,label='Group'))
        help_str = "All pupil nodes of one group share a Master clock."
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('sync status',getter=self.sync_status_info,setter=lambda _: _))
        # self.menu[-1].read_only = True
        self.group_menu = ui.Growing_Menu('Other Pupils')
        self.menu.append(self.group_menu)
        self.g_pool.sidebar.append(self.menu)
        self.update_gui()

    def update_gui(self):
        if self.group_menu:
            self.group_menu.elements[:] = []
            for uuid in self.group_members.keys():
                self.group_menu.append(ui.Info_Text("%s"%self.group_members[uuid]))

    def set_name(self,new_name):
        self.name = new_name
        if self.thread_pipe:
            self.thread_pipe.send(exit_thread)
            while self.thread_pipe:
                sleep(.01)
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)

    def set_group(self,new_name):
        self.group = new_name
        if self.thread_pipe:
            self.thread_pipe.send(exit_thread)
            while self.thread_pipe:
                sleep(.01)
        self.group_members = {}
        self.update_gui()
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)


    def close(self):
        self.alive = False

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def thread_loop(self,context,pipe):
        n = Pyre(self.name)
        n.join(self.group)
        n.start()
        poller = zmq.Poller()
        poller.register(pipe, zmq.POLLIN)
        poller.register(n.socket(), zmq.POLLIN)


        front,back = zhelper.zcreate_pipe(context)
        poller.register(back, zmq.POLLIN)
        def wake_up():
            #on app close this timer calls a closed socket. We simply catch it here.

            try:
                front.send('wake_up')
            except Exception as e:
                logger.debug('Orphaned timer thread raised error: %s'%e)

        t = Timer(self.time_sync_announce_interval, wake_up)
        t.daemon = True
        t.start()

        while(True):
            try:
                #this should not fail but it does sometimes. We need to clean this out.
                # I think we are not treating sockets correclty as they are not thread-save.
                items = dict(poller.poll())
            except zmq.ZMQError:
                logger.warning('Socket fail.')
                continue

            if back in items and items[back] == zmq.POLLIN:
                back.recv()
                #timeout events are used for pupil sync.
                #annouce masterhood every interval time:
                if isinstance(self.sync_node,Clock_Sync_Master):
                    n.shouts(self.group, sync_time_master_announce+"%s"%self.clock_master_worthiness()+msg_delimeter+'%s'%self.sync_node.port)

                # synced slave: see if we should become master if we dont hear annoncement within time.
                elif isinstance(self.sync_node,Clock_Sync_Follower) and not self.sync_node.offset_remains:
                    if self.get_monotonic_time()-self.last_master_announce > self.time_sync_wait_interval_short:
                        self.sync_node.terminate()
                        self.sync_node = Clock_Sync_Master(time_fn=self.get_time)
                        n.shouts(self.group, sync_time_master_announce+"%s"%self.clock_master_worthiness()+msg_delimeter+'%s'%self.sync_node.port)

                # unsynced slave or none should wait longer but eventually take over
                elif self.get_monotonic_time()-self.last_master_announce > self.time_sync_wait_interval_long:
                    if self.sync_node:
                        self.sync_node.terminate()
                    self.sync_node = Clock_Sync_Master(time_fn=self.get_time)
                    n.shouts(self.group, sync_time_master_announce+"%s"%self.clock_master_worthiness()+msg_delimeter+'%s'%self.sync_node.port)

                t = Timer(self.time_sync_announce_interval, wake_up)
                t.daemon = True
                t.start()


            if pipe in items and items[pipe] == zmq.POLLIN:
                message = pipe.recv()
                # message to quit
                if message.decode('utf-8') == exit_thread:
                    break
                else:
                    logger.debug("Shout '%s' to '%s' " %(message,self.group))
                    n.shouts(self.group, message)
            if n.socket() in items and items[n.socket()] == zmq.POLLIN:
                cmds = n.recv()
                msg_type = cmds.pop(0)
                msg_type = msg_type.decode('utf-8')
                if msg_type == "SHOUT":
                    uuid,name,group,msg = cmds
                    logger.debug("'%s' shouts '%s'."%(name,msg))
                    self.handle_msg(uuid,name,msg,n)

                elif msg_type == "WHISPER":
                    uuid,name,msg = cmds
                    logger.debug("'%s/' whispers '%s'."%(name,msg))
                    self.handle_msg_whisper(uuid,name,msg,n)

                elif msg_type == "JOIN":
                    uuid,name,group = cmds
                    if group == self.group:
                        self.group_members[uuid] = name
                        self.update_gui()

                elif msg_type == "EXIT":
                    uuid,name = cmds
                    try:
                        del self.group_members[uuid]
                    except KeyError:
                        pass
                    else:
                        self.update_gui()

                # elif msg_type == "LEAVE":
                #     uuid,name,group = cmds
                # elif msg_type == "ENTER":
                #     uuid,name,headers,ip = cmds
                #     logger.warning((uuid,'name',headers,ip))
            else:
                pass

        logger.debug('thread_loop closing.')

        self.thread_pipe = None
        n.stop()


    def handle_msg(self,uuid,name,msg,n):

        #Clock Sync master announce logic
        if sync_time_master_announce in msg:

            self.last_master_announce = self.get_monotonic_time()

            worthiness,port = msg.replace(sync_time_master_announce,'').split(msg_delimeter)
            foreign_master_worthiness = float(worthiness)
            foreign_master_port = int(port)
            forein_master_uuid = UUID(bytes=uuid)
            foreign_master_address = n.peer_address(forein_master_uuid)
            foreign_master_ip = foreign_master_address.split('//')[-1].split(':')[0]  # tcp://10.0.1.68:59149

            if isinstance(self.sync_node,Clock_Sync_Master):
                # who should yield?
                if not self.was_adjusted() and foreign_master_adjusted:
                    should_yield = True
                elif self.nerver_adjusted() and (not foreign_master_adjusted):
                    should_yield = False
                else:
                    if n.uuid().int > forein_master_uuid.int:
                        self.should_yield = False
                    else:
                        self.should_yield = True

                if should_yield:
                    logger.warning("Yield Clock_Sync_Master to %s@%s"%(name,new_master))
                    self.sync_node.stop()
                    self.sync_node = Clock_Sync_Follower(foreign_master_ip,port=foreign_master_port,interval=10,time_fn=self.get_time,jump_fn=self.jump_time,slew_fn=self.set_time)
                else:
                    logger.warning("Dominate as Clock_Sync_Master")
                    n.shouts(self.group, sync_time_master_announce+"%s"%self.clock_master_worthiness()+msg_delimeter+'%s'%self.sync_node.port)

            elif isinstance(self.sync_node,Clock_Sync_Follower):
                self.sync_node.host = foreign_master_ip
                self.sync_node.port = foreign_master_port
            else:
                self.sync_node = Clock_Sync_Follower(foreign_master_ip,port=foreign_master_port,interval=10,time_fn=self.get_time,jump_fn=self.jump_time,slew_fn=self.set_time)
                logger.debug("Clock synced with %s"%foreign_master_ip)

        elif start_rec in msg :
            session_name = msg.replace(start_rec,'')
            self.notify_all({'subject':'rec_should_start','session_name':session_name,'network_propagate':False})
        elif stop_rec in msg:
            self.notify_all({'subject':'rec_should_stop','network_propagate':False})
        elif start_cal in msg:
            self.notify_all({'subject':'cal_should_start'})
        elif stop_cal in msg:
            self.notify_all({'subject':'cal_should_stop'})
        elif user_event in msg:
            payload = msg.replace(user_event,'')
            user_event_name,timestamp = payload.split('@')
            self.notify_all({'subject':'remote_user_event','user_event_name':user_event_name,'timestamp':float(timestamp),'network_propagate':False,'sender':name,'received_timestamp':self.g_pool.capture.get_timestamp()})

    def handle_msg_whisper(self,peer,name,msg,node):
        logger.warning('%s %s %s %s'%(peer,name,msg,node))


    def on_notify(self,notification):
        # if we get a rec event that was not triggered though pupil_sync it will carry network_propage=True
        # then we should tell other Pupils to mirror this action
        # this msg has come because rec was triggered through pupil sync,
        # we dont need to echo this action out again.
        # otherwise we create a feedback loop and bad things happen.
        if notification['subject'] == 'rec_started' and notification['network_propagate']:
            self.thread_pipe.send(start_rec+notification['session_name'])
        elif notification['subject'] == 'rec_stopped' and notification['network_propagate']:
            self.thread_pipe.send(stop_rec)

        #userevents are also sycronized
        elif notification['subject'] == 'local_user_event':
            self.thread_pipe.send('%s%s@%s'%(user_event,notification['user_event_name'],notification['timestamp']))


    def get_init_dict(self):
        return {'name':self.name,'group':self.group}

    def cleanup(self):
        """gets called when the plugin get terminated.
           This happens either volunatily or forced.
        """
        if self.sync_node:
            self.sync_node.terminate()
        self.deinit_gui()
        self.thread_pipe.send(exit_thread)
        while self.thread_pipe:
            sleep(.01)
        self.context.destroy()

