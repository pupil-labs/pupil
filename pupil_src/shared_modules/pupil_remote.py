'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from time import sleep
import socket
import audio
import zmq
import zmq_tools
from pyre import zhelper
from pyglui import ui
from plugin import Plugin
import logging
logger = logging.getLogger(__name__)


class Pupil_Remote(Plugin):
    """Pupil Remote plugin

    Send simple string messages to control Pupil Capture functions:
        'R' start recording with auto generated session name
        'R rec_name' start recording and name new session name: rec_name
        'r' stop recording
        'C' start currently selected calibration
        'c' stop currently selected calibration
        'T 1234.56' Timesync: make timestamps count form 1234.56 from now on.
        't' get pupil capture timestamp returns a float as string.


        # IPC Backbone communication
        'PUB_PORT' return the current pub port of the IPC Backbone
        'SUB_PORT' return the current sub port of the IPC Backbone

    Mulitpart messages conforming to pattern:
        part1: 'notify.' part2: a msgpack serialized dict with at least key 'subject':'my_notification_subject'
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

    Attributes:
        address (str): Remote host address
        alive (bool): See plugin.py
        context (zmq.Context): zmq context
        menu (ui.Growing_Menu): Sidebar menu
        order (float): See plugin.py
        thread_pipe (zmq.Socket): Pipe for background communication
    """
    def __init__(self, g_pool, port="50020", host="*", use_primary_interface=True):
        super().__init__(g_pool)
        self.order = .01  # excecute first
        self.context = g_pool.zmq_ctx
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)

        self.use_primary_interface = use_primary_interface
        assert type(host) == str
        assert type(port) == str
        self.host = host
        self.port = port

        self.start_server('tcp://{}:{}'.format(host, port))
        self.menu = None

    def start_server(self, new_address):
        self.thread_pipe.send_string('Bind', flags=zmq.SNDMORE)
        self.thread_pipe.send_string(new_address)
        response = self.thread_pipe.recv_string()
        msg = self.thread_pipe.recv_string()
        if response == 'Bind OK':
            host, port = msg.split(':')
            self.host = host
            self.port = port
            return

        # fail logic
        logger.error(msg)

        # for service we shut down
        if self.g_pool.app == 'service':
            audio.say("Error: Port already in use.")
            self.notify_all({'subject': 'service_process.should_stop'})
            return

        # for capture we try to bind to a arbitrary port on the first external interface
        else:
            self.thread_pipe.send_string('Bind', flags=zmq.SNDMORE)
            self.thread_pipe.send_string('tcp://*:*')
            response = self.thread_pipe.recv_string()
            msg = self.thread_pipe.recv_string()
            if response == 'Bind OK':
                host, port = msg.split(':')
                self.host = host
                self.port = port
            else:
                logger.error(msg)
                raise Exception("Could not bind to port")

    def stop_server(self):
        self.thread_pipe.send_string('Exit')
        while self.thread_pipe:
            sleep(.1)

    def init_gui(self):
        self.menu = ui.Growing_Menu('Pupil Remote')
        self.menu.collapsed = True
        self.g_pool.sidebar.append(self.menu)
        self.update_menu()

    def update_menu(self):

        del self.menu.elements[:]

        def close():
            self.alive = False

        def set_iface(use_primary_interface):
            self.use_primary_interface = use_primary_interface
            self.update_menu()

        if self.use_primary_interface:
            def set_port(new_port):
                new_address = 'tcp://*:'+new_port
                self.start_server(new_address)
                self.update_menu()

            try:
                ip = socket.gethostbyname(socket.gethostname())
            except:
                ip = 'Your external ip'

        else:
            def set_address(new_address):
                if new_address.count(":") != 1:
                    logger.error("address format not correct")
                    return
                self.start_server('tcp://'+new_address)
                self.update_menu()

        help_str = 'Pupil Remote using ZeroMQ REQ REP scheme.'
        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Switch('use_primary_interface', self, setter=set_iface, label="Use primary network interface"))
        if self.use_primary_interface:
            self.menu.append(ui.Text_Input('port', self, setter=set_port, label='Port'))
            self.menu.append(ui.Info_Text('Connect locally:   "tcp://127.0.0.1:{}"'.format(self.port)))
            self.menu.append(ui.Info_Text('Connect remotely: "tcp://{}:{}"'.format(ip, self.port)))
        else:
            self.menu.append(ui.Text_Input('host', setter=set_address, label='Address',
                                           getter=lambda: '{}:{}'.format(self.host, self.port)))
            self.menu.append(ui.Info_Text('Bound to: "tcp://{}:{}"'.format(self.host, self.port)))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def thread_loop(self, context, pipe):
        poller = zmq.Poller()
        ipc_pub = zmq_tools.Msg_Dispatcher(context, self.g_pool.ipc_push_url)
        poller.register(pipe, zmq.POLLIN)
        remote_socket = None

        while True:
            items = dict(poller.poll())
            if pipe in items:
                cmd = pipe.recv_string()
                if cmd == 'Exit':
                    break
                elif cmd == 'Bind':
                    new_url = pipe.recv_string()
                    if remote_socket:
                        poller.unregister(remote_socket)
                        remote_socket.close(linger=0)
                    try:
                        remote_socket = context.socket(zmq.REP)
                        remote_socket.bind(new_url)
                    except zmq.ZMQError as e:
                        remote_socket = None
                        pipe.send_string("Error", flags=zmq.SNDMORE)
                        pipe.send_string("Could not bind to Socket: {}. Reason: {}".format(new_url, e))
                    else:
                        pipe.send_string("Bind OK", flags=zmq.SNDMORE)
                        # `.last_endpoint` is already of type `bytes`
                        pipe.send(remote_socket.last_endpoint.replace(b"tcp://", b""))
                        poller.register(remote_socket, zmq.POLLIN)
            if remote_socket in items:
                self.on_recv(remote_socket, ipc_pub)

        self.thread_pipe = None

    def on_recv(self, socket, ipc_pub):
        msg = socket.recv_string()
        if msg.startswith('notify'):
            try:
                payload = zmq_tools.serializer.loads(socket.recv(flags=zmq.NOBLOCK), encoding='utf-8')
                payload['subject']
            except Exception as e:
                response = 'Notification mal-formatted or missing: {}'.format(e)
            else:
                ipc_pub.notify(payload)
                response = 'Notification recevied.'
        elif msg == 'SUB_PORT':
            response = self.g_pool.ipc_sub_url.split(':')[-1]
        elif msg == 'PUB_PORT':
            response = self.g_pool.ipc_pub_url.split(':')[-1]
        elif msg[0] == 'R':
            try:
                ipc_pub.notify({'subject': 'recording.should_start', 'session_name': msg[2:]})
                response = 'OK'
            except IndexError:
                response = 'Recording command mal-formatted.'
        elif msg[0] == 'r':
            ipc_pub.notify({'subject': 'recording.should_stop'})
            response = 'OK'
        elif msg == 'C':
            ipc_pub.notify({'subject': 'calibration.should_start'})
            response = 'OK'
        elif msg == 'c':
            ipc_pub.notify({'subject': 'calibration.should_stop'})
            response = 'OK'
        elif msg[0] == 'T':
            try:
                target = float(msg[2:])
            except:
                response = "'{}' cannot be converted to float.".format(msg[2:])
            else:
                raw_time = self.g_pool.get_now()
                self.g_pool.timebase.value = raw_time-target
                response = 'Timesync successful.'
        elif msg[0] == 't':
            response = repr(self.g_pool.get_timestamp())
        else:
            response = 'Unknown command.'
        socket.send_string(response)

    def on_notify(self, notification):
        """send simple string messages to control application functions.

        Emits notifications:
            ``recording.should_start``
            ``recording.should_stop``
            ``calibration.should_start``
            ``calibration.should_stop``
            Any other notification received though the reqrepl port.
        """
        pass

    def get_init_dict(self):
        return {'port': self.port, 'host': self.host, 'use_primary_interface': self.use_primary_interface}

    def cleanup(self):
        """gets called when the plugin get terminated.
           This happens either voluntarily or forced.
        """
        self.stop_server()
        self.deinit_gui()
