'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from time import sleep
from plugin import Plugin
from pyglui import ui
import zmq
import zmq_tools
from pyre import zhelper
import logging
logger = logging.getLogger(__name__)


class Pupil_Remote(Plugin):
    """pupil remote plugin
    send simple string messages to control Pupil Capture functions:

    'R' start recording with auto generated session name
    'R rec_name' start recording and name new session name: rec_name
    'r' stop recording
    'C' start currently selected calibration
    'c' stop currently selected calibration
    'T 1234.56' Timesync: make timestamps count form 1234.56 from now on.
    't' get pupil capture timestamp returns a float as string.


    #IPC Backbone communication
    'PUB_PORT' return the current pub port of the IPC Backbone
    'SUB_PORT' return the current sub port of the IPC Backbone

    mulitpart messages conforming to pattern:
    part1: 'notify.' part2: json encoded dict with at least  key 'subject':'my_notification_subject'
    will be forwared to the Pupil IPC Backbone.


    A example script for talking with pupil remote below:
    import zmq
    from time import sleep,time
    context =  zmq.Context()
    socket = context.socket(zmq.REQ)
    # set your ip here
    socket.connect('tcp://192.168.1.100:50020')
    t= time()
    socket.send('t')
    print socket.recv()
    print 'Round trip command delay:', time()-t
    print 'If you need continous syncing and/or less latency look at pupil_sync.'
    sleep(1)
    socket.send('R')
    print socket.recv()
    sleep(5)
    socket.send('r')
    print socket.recv()
    """
    def __init__(self, g_pool,address="tcp://*:50020"):
        super(Pupil_Remote, self).__init__(g_pool)
        self.order = .01 #excecute first
        self.context = g_pool.zmq_ctx
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)
        self.address = address
        self.start_server(address)
        self.menu = None


    def start_server(self,new_address):
        self.thread_pipe.send('Bind')
        self.thread_pipe.send(new_address)
        response = self.thread_pipe.recv()
        if response == 'Bind OK':
            self.address = new_address
        else:
            logger.error(response)
            if self.g_pool.app == 'service':
                self.notify_all({'subject':'service_process.should_stop'})
            else:
                self.address = ''

    def stop_server(self):
        self.thread_pipe.send('Exit')
        while self.thread_pipe:
            sleep(.1)

    def init_gui(self):

        def close():
            self.alive = False

        help_str = 'Pupil Remote using ZeroMQ REQ REP scheme.'
        self.menu = ui.Growing_Menu('Pupil Remote')
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('address',self,setter=self.start_server,label='Address'))
        self.g_pool.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None


    def thread_loop(self,context,pipe):
        poller = zmq.Poller()
        ipc_pub = zmq_tools.Msg_Dispatcher(context,self.g_pool.ipc_push_url)
        poller.register(pipe, zmq.POLLIN)
        remote_socket = None

        while True:
            items = dict(poller.poll())
            if items.get(pipe,None) == zmq.POLLIN:
                cmd = pipe.recv()
                if cmd == 'Exit':
                    break
                elif cmd == 'Bind':
                    new_url = pipe.recv()
                    if remote_socket:
                        poller.unregister(remote_socket)
                        remote_socket.close(linger=0)
                    try:
                        remote_socket = context.socket(zmq.REP)
                        remote_socket.bind(new_url)
                    except zmq.ZMQError as e:
                        remote_socket = None
                        pipe.send("Could not bind to Socket: %s. Reason: %s"%(new_url,e))
                    else:
                        pipe.send("Bind OK")
                        poller.register(remote_socket)
            if items.get(remote_socket,None) == zmq.POLLIN:
                self.on_recv(remote_socket,ipc_pub)

        self.thread_pipe = None

    def on_recv(self,socket,ipc_pub):
        msg = socket.recv()
        if msg.startswith('notify'):
            try:
                payload = zmq_tools.json.loads(socket.recv(flags=zmq.NOBLOCK))
                payload['subject']
            except Exception as e:
                response = 'Notification mal-formatted or missing: %s'%e
            else:
                ipc_pub.notify(payload)
                response = 'Notification recevied.'
        elif msg == 'SUB_PORT':
            response = self.g_pool.ipc_sub_url.split(':')[-1]
        elif msg == 'PUB_PORT':
            response = self.g_pool.ipc_pub_url.split(':')[-1]
        elif msg[0] == 'R':
            ipc_pub.notify({'subject':'recording.should_start','session_name':msg[2:]})
            response = 'OK'
        elif msg[0] == 'r':
            ipc_pub.notify({'subject':'recording.should_stop'})
            response = 'OK'
        elif msg == 'C':
            ipc_pub.notify({'subject':'calibration.should_start'})
            response = 'OK'
        elif msg == 'c':
            ipc_pub.notify({'subject':'calibration.should_stop'})
            response = 'OK'
        elif msg[0] == 'T':
            try:
                target = float(msg[2:])
            except:
                response = "'%s' cannot be converted to float."%msg[2:]
            else:
                raw_time = self.g_pool.get_now()
                self.g_pool.timebase.value = raw_time-target
                response = 'Timesync successful.'
        elif msg[0] == 't':
            response = str(self.g_pool.get_timestamp())
        else:
            response = 'Unknown command.'
        socket.send(response)


    def get_init_dict(self):
        return {'address':self.address}

    def cleanup(self):
        """gets called when the plugin get terminated.
           This happens either voluntarily or forced.
        """
        self.stop_server()
        self.deinit_gui()




