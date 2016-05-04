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
from pyre import zhelper
import logging
logger = logging.getLogger(__name__)

exit_thread = "EXIT_THREAD".encode('utf_8')


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
        self.context = zmq.Context()
        self.thread_pipe = None
        self.address = 'not set'
        self.start_server(address)


    def start_server(self,new_address):
        if self.thread_pipe:
            self.stop_server()
        try:
            socket = self.context.socket(zmq.REP)
            socket.bind(new_address)
        except zmq.ZMQError as e:
            logger.error("Could not bind to Socket: %s. Reason: %s"%(new_address,e))
        else:
            self.address = new_address
            self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop,socket)


    def stop_server(self):
        self.thread_pipe.send(exit_thread)
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


    def thread_loop(self,context,pipe,socket):
        poller = zmq.Poller()
        poller.register(pipe, zmq.POLLIN)
        poller.register(socket, zmq.POLLIN)

        while True:
            try:
                #this should not fail but it does sometimes. We need to clean this out.
                # I think we are not treating sockets correclty as they are not thread-safe.
                items = dict(poller.poll())
            except zmq.ZMQError:
                logger.warning('Socket fail.')
            else:
                if items.get(pipe,None) == zmq.POLLIN:
                    message = pipe.recv()
                    if message.decode('utf-8') == exit_thread:
                        break
                if items.get(socket,None) == zmq.POLLIN:
                    self.on_recv(socket)

        socket.close()
        self.thread_pipe = None

    def on_recv(self,socket):
        msg = socket.recv()
        if msg[0] == 'R':
            self.notify_all({'subject':'should_start_recording','session_name':msg[2:]})
            response = 'started recording'
        elif msg[0] == 'r':
            self.notify_all({'subject':'should_stop_recording'})
            response = 'stopped recording'
        elif msg == 'C':
            self.notify_all({'subject':'should_start_calibration'})
            response = 'started calibration'
        elif msg == 'c':
            self.notify_all({'subject':'should_stop_calibration'})
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
        self.context.destroy()




