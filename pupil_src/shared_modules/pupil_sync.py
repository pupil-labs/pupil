'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from plugin import Plugin

from pyglui import ui
import zmq
from pyre import Pyre
from pyre import zhelper
import uuid
from time import sleep
import logging
logger = logging.getLogger(__name__)


start_rec = "START_REC:"
stop_rec = "STOP_REC:"
start_cal = "START_CAL"
stop_cal = "STOP_CAL"
sync_time_init = "SYNC_INIT:"
sync_time_request = "SYNC_REQ:"
sync_time_reply = "SYNC_RPL:"
user_event = "USREVENT:"


# Pipe signals:
exit_thread = "EXIT_THREAD".encode('utf_8')
init_master_sync = "INIT_SYNC".encode('utf_8')

'''
Time synchonization scheme:

the initializing node is called master, others just nodes.
the new timebase is the one the master currenlty runs on:
the master then starts whispering to each node:
for node in other_nodes:
    send time req to node
    now the node takes over command:
    node send local time  (t0) to master
    master returns with massage of own time stamp (t1)
    Upon receipt by master, node take time t2 then: latency: t2-t0 target time = t1+latency/2
    node sets timebase to target time.
    for x in range(5)
        measure t0,t1,t2, latency, offset = local_time -target_time
    calulate mean, variance of latency
        take avg offset of measurements that are not outliers
    apply offset to local time

'''




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

        self.context = zmq.Context()
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)



        #variables for the time sync logic
        self.sync_master = None
        self.sync_data = []
        self.sync_to_collect = 0
        self.sync_nodes = []
        self.timeout = None



    def init_gui(self):
        help_str = "Synchonize behaviour of Pupil captures across the local network."
        self.menu = ui.Growing_Menu('Pupil Sync')
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('name',self,setter=self.set_name,label='Name'))
        self.menu.append(ui.Text_Input('group',self,setter=self.set_group,label='Group'))
        help_str = "Before starting a recording. Make sure to sync the timebase of all Pupils to one master Pupil by clicking the bottom below. This will apply this Pupil's timebase to all of its group."
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Button('sync time for all Pupils',self.set_sync))
        self.group_menu = ui.Growing_Menu('Other Pupils')
        self.menu.append(self.group_menu)
        self.g_pool.sidebar.append(self.menu)
        self.update_gui()

    def update_gui(self):
        if self.group_menu:
            self.group_menu.elements[:] = []
            for uid in self.group_members.keys():
                self.group_menu.append(ui.Info_Text("%s"%self.group_members[uid]))

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

    def set_sync(self):
        self.sync_master = self
        self.sync_nodes = self.group_members.keys()
        self.thread_pipe.send(init_master_sync)

    def set_timebase(self,new_time):
        self.g_pool.timebase.value = float(new_time)
        logger.debug("New timebase set to %s all timestamps will count from here now."%self.g_pool.timebase.value)


    def ok_to_set_timebase(self):
        ok_to_change = True
        for p in self.g_pool.plugins:
            if p.class_name == 'Recorder':
                if p.running:
                    ok_to_change = False
                    logger.warning("Request to change timebase during recording ignored. Turn off recording first.")
        return ok_to_change

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
        logger.debug(n.socket())
        poller.register(n.socket(), zmq.POLLIN)

        while(True):
            try:
                #this should not fail but it does sometimes. We need to clean this out.
                # I think we are not treating sockets correclty as they are not thread-save.
                items = dict(poller.poll(self.timeout))
            except zmq.ZMQError:
                logger.warning('Socket fail.')
            # print(n.socket(), items)
            if pipe in items and items[pipe] == zmq.POLLIN:
                message = pipe.recv()
                # message to quit
                if message.decode('utf-8') == exit_thread:
                    break
                elif message.decode('utf-8') == init_master_sync:
                    self.timeout = 3000
                else:
                    logger.debug("Emitting to '%s' to '%s' " %(message,self.group))
                    n.shouts(self.group, message)
            if n.socket() in items and items[n.socket()] == zmq.POLLIN:
                cmds = n.recv()
                msg_type = cmds.pop(0)
                msg_type = msg_type.decode('utf-8')
                if msg_type == "SHOUT":
                    uid,name,group,msg = cmds
                    logger.debug("'%s' shouts '%s'."%(name,msg))
                    self.handle_msg(name,msg)

                elif msg_type == "WHISPER":
                    uid,name,msg = cmds
                    logger.debug("'%s/' whispers '%s'."%(name,msg))
                    self.handle_msg_whisper(uid,name,msg,n)

                elif msg_type == "JOIN":
                    uid,name,group = cmds
                    if group == self.group:
                        self.group_members[uid] = name
                        self.update_gui()

                elif msg_type == "EXIT":
                    uid,name = cmds
                    try:
                        del self.group_members[uid]
                    except KeyError:
                        pass
                    else:
                        self.update_gui()

                # elif msg_type == "LEAVE":
                #     uid,name,group = cmds
                # elif msg_type == "ENTER":
                #     uid,name,headers,ip = cmds
                #     logger.warning((uid,'name',headers,ip))

            elif not items:
                #timeout events are used for pupil sync.
                if self.sync_master is self:
                    if self.sync_nodes:
                        node_uid = self.sync_nodes.pop(0)
                        logger.info("Synchonizing node %s"%self.group_members[node_uid])
                        n.whispers(uuid.UUID(bytes=node_uid),sync_time_init)
                    else:
                        self.timeout = None
                        self.sync_master = None
                        logger.info("All other Pupil nodes are sycronized.")
                elif self.sync_master:
                    t0 = self.g_pool.capture.get_timestamp()
                    n.whispers(uuid.UUID(bytes=self.sync_master),sync_time_request+'%s'%t0)

            else:
                pass

        logger.debug('thread_loop closing.')
        self.thread_pipe = None
        n.stop()


    def handle_msg(self,name,msg):
        if start_rec in msg :
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

        #when acting as master during sync all we need to do is reply to requests.
        if sync_time_request in msg:
            t0 = msg.replace(sync_time_request,'')
            node.whispers(uuid.UUID(bytes=peer),sync_time_reply+t0+':'+"%s"%self.g_pool.capture.get_timestamp())


        #as node during sync we need to note reply times measure and set our timebase.
        elif sync_time_reply in msg:
            t2 = self.g_pool.capture.get_timestamp()
            t0_t1 = msg.replace(sync_time_reply,'')
            t0,t1 = (float(t) for t in t0_t1.split(':'))

            logger.info("Round trip sync latency %sms "%(t2-t0))

            if self.sync_to_collect == 5:
                #first run, we do a coarse adjustment
                new_time = t1 + ((t2-t0)/2.)
                if self.ok_to_set_timebase:
                    self.set_timebase(new_time)

            elif self.sync_to_collect > 0:
                #collect 4 more samples
                self.sync_data.append((t0,t1,t2))


            elif self.sync_to_collect == 0:
                self.timeout = None

            self.sync_to_collect -= 1


        elif sync_time_init in msg:
            self.sync_master = peer
            self.sync_data = []
            self.sync_to_collect = 5
             #from now on zmq.poller will timeout when the lines are clear and we use this to send sync messages.
            self.timeout = 200



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
        self.deinit_gui()
        self.thread_pipe.send(exit_thread)
        while self.thread_pipe:
            sleep(.01)
        self.context.destroy()

