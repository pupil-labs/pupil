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
import struct
import audio
import zmq
import zmq_tools
from pyre import zhelper
from pyglui import ui
from plugin import Plugin
import logging
logger = logging.getLogger(__name__)


__version__ = 1

class UDP_Backend(Plugin):
    """UDP_Backend
    """
    icon_chr = chr(0xe307)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, port="50021", host="", use_primary_interface=True):
        super().__init__(g_pool)
        self.order = .01  # excecute first
        self.context = g_pool.zmq_ctx
        self.thread_pipe = zhelper.zthread_fork(self.context, self.thread_loop)

        self.use_primary_interface = use_primary_interface
        assert type(host) == str
        assert type(port) == str
        self.host = host
        self.port = port

        self.start_server('{}:{}'.format(host, port))
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
        raise Exception("Could not bind to port")

    def stop_server(self):
        self.thread_pipe.send_string('Exit')
        while self.thread_pipe:
            sleep(.1)

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'UDP Backend'
        self.update_menu()

    def deinit_ui(self):
        self.remove_menu()

    def update_menu(self):

        del self.menu.elements[:]

        def set_iface(use_primary_interface):
            self.use_primary_interface = use_primary_interface
            self.update_menu()

        if self.use_primary_interface:
            def set_port(new_port):
                new_address = ':'+new_port
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
                self.start_server(new_address)
                self.update_menu()

        help_str = 'The UDP backend allows remote control and gaze relay via udp.'
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Switch('use_primary_interface', self, setter=set_iface, label="Use primary network interface"))
        if self.use_primary_interface:
            self.menu.append(ui.Text_Input('port', self, setter=set_port, label='Port'))
            self.menu.append(ui.Info_Text('Connect locally:   "127.0.0.1:{}"'.format(self.port)))
            self.menu.append(ui.Info_Text('Connect remotely: "{}:{}"'.format(ip, self.port)))
        else:
            self.menu.append(ui.Text_Input('host', setter=set_address, label='Address',
                                           getter=lambda: '{}:{}'.format(self.host, self.port)))
            self.menu.append(ui.Info_Text('Bound to: "{}:{}"'.format(self.host, self.port)))


    def thread_loop(self, context, pipe):
        poller = zmq.Poller()
        ipc_pub = zmq_tools.Msg_Dispatcher(context, self.g_pool.ipc_push_url)
        ipc_sub = zmq_tools.Msg_Receiver(context, self.g_pool.ipc_sub_url)
        poller.register(pipe, zmq.POLLIN)
        poller.register(ipc_sub.socket, zmq.POLLIN)
        remote_socket = None
        gaze_receiver = None

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
                        remote_socket.close()
                    try:
                        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        remote_socket.setblocking(False)
                        addr, port = new_url.split(':')
                        socket_addr = (addr, int(port))
                        remote_socket.bind(socket_addr)
                    except OSError as e:
                        remote_socket = None
                        pipe.send_string("Error", flags=zmq.SNDMORE)
                        pipe.send_string("Could not bind to Socket: {}. Reason: {}".format(new_url, e))
                    else:
                        pipe.send_string("Bind OK", flags=zmq.SNDMORE)
                        # TODO: get addr
                        pipe.send_string(new_url)
                        poller.register(remote_socket, zmq.POLLIN)
            # if remote_socket in items:
            gaze_receiver = self.on_recv(remote_socket, ipc_pub, gaze_receiver)

            if ipc_sub in items:
                while ipc_sub.new_data:
                    # we should only receive gaze data
                    payload = ipc_sub.recv()[1]
                    if remote_socket is not None and gaze_receiver is not None:
                        if '2d' in payload['method']:
                            data = b'g%s'%struct.pack('ff', *payload['norm_pos'])
                        elif '3d' in payload['method']:
                            data = b'G%s'%struct.pack('fff', *payload['gaze_point_3d'])
                        else:
                            data = b'EgError while relaying gaze'
                        remote_socket.sendto(data, gaze_receiver)

        self.thread_pipe = None

    def on_recv(self, socket, ipc_pub, gaze_receiver):
        try:
            byte_msg, sender = socket.recvfrom(1024)
        except OSError:
            return
        print(byte_msg, sender)
        if byte_msg[:1] == b'R':  # reference point
            try:
                # relay reference point
                ipc_pub.socket.send_string('notify.calibration.add_ref_data', flags=zmq.SNDMORE)
                ipc_pub.socket.send(byte_msg[1:])
            except Exception as e:
                # respond with <error byte><headercode byte>[<byte>, ...]
                response = b'ERReference point mal-formatted or missing: %s'%str(e).encode()
            else:
                response = b'0R'
        elif byte_msg[:1] == b'S':
            gaze_receiver = sender
            response = b'0S'
        elif byte_msg[:1] == b's':
            gaze_receiver = None
            response = b'0s'
        elif byte_msg[:1] == b'M':
            mode = struct.unpack('B', byte_msg[1])
            # todo: start both eyes, set mode
            response = b'0M'
        elif byte_msg[:1] == b'C':
            ipc_pub.notify({'subject': 'calibration.should_start'})
            response = b'0C'
        elif byte_msg[:1] == b'c':
            ipc_pub.notify({'subject': 'calibration.should_stop'})
            response = b'0c'
        elif byte_msg[:1] == b'T':
            try:
                target = struct.unpack('f', byte_msg[1])
            except IndexError:
                response = b'ETTimestamp required'
            except:
                response = b"ET'%s' cannot be converted to float."%(byte_msg[1])
            else:
                raw_time = self.g_pool.get_now()
                self.g_pool.timebase.value = raw_time-target
                response = b'0T'
        elif byte_msg[:1] == b'V':
            response = b'0V%s'%bytes(__version__)
        else:
            response = b'EEUnknown command. "%s"'%byte_msg
        socket.sendto(response, sender)
        return gaze_receiver

    def get_init_dict(self):
        return {'port': self.port, 'host': self.host, 'use_primary_interface': self.use_primary_interface}

    def cleanup(self):
        """gets called when the plugin get terminated.
           This happens either voluntarily or forced.
        """
        self.stop_server()
