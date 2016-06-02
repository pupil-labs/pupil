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

    Pupil Remote is the simplistic version of Pupil Sync:
    Not as good but the protocol is dead simple.
    A example script for talking with pupil remote below:
    import zmq
    from time import sleep,time
    context =  zmq.Context()
    socket = context.socket(zmq.REQ)
    # set your ip here
    socket.connect('tcp://192.168.1.100:50020')
    t= time()
    socket.send('T 0.0')
    print socket.recv()
    print 'Round trip command delay:', time()-t
    print 'If you need continous syncing and less latency look at pupil_sync.'
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
        self.start_server(address)


    def start_server(self,new_address):
        self.thread_pipe.send('Bind')
        self.thread_pipe.send(new_address)
        response = self.thread_pipe.recv()
        if response == 'Bind OK':
            self.address = new_address
        else:
            logger.error(response)

    def stop_server(self):
        self.thread_pipe.send('Exit')
        while self.thread_pipe:
            sleep(.01)

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
        ipc_pub = zmq_tools.Msg_Dispatcher(context,self.g_pool.ipc_pub_url)
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
                        pipe.send("Could not bind to Socket: %s. Reason: %s"%(new_address,e))
                    else:
                        pipe.send("Bind OK")
                        poller.register(remote_socket)
            if items.get(remote_socket,None) == zmq.POLLIN:
                self.on_recv(remote_socket,ipc_pub)

        if remote_socket:
            remote_socket.close(linger=0)
            del ipc_pub

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
                response = 'Notifaction recevied.'
        elif msg == 'SUB_URL':
            response = self.g_pool.ipc_sub_url
        elif msg == 'PUB_URL':
            response = self.g_pool.ipc_pub_url
        elif msg[0] == 'R':
            ipc_pub.notify({'subject':'should_start_recording','session_name':msg[2:]})
            response = 'started recording'
        elif msg[0] == 'r':
            ipc_pub.notify({'subject':'should_stop_recording'})
            response = 'stopped recording'
        elif msg == 'C':
            ipc_pub.notify({'subject':'should_start_calibration'})
            response = 'started calibration'
        elif msg == 'c':
            ipc_pub.notify({'subject':'should_stop_calibration'})
            response = 'stopped calibration'
        elif msg[0] == 'T':
            try:
                target = float(msg[2:])
            except:
                response = "'%s' cannot be converted to float."%msg[2:]
            else:
                raw_time = self.g_pool.capture.get_now()
                self.g_pool.timebase.value = raw_time-target
                response = 'Timesync successful.'
        elif msg[0] == 't':
            response = str(self.g_pool.capture.get_timestamp())
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




