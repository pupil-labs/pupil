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
import zmq
import zmq_tools
from pyre import zhelper
from pyglui import ui
from plugin import Plugin
import logging
logger = logging.getLogger(__name__)


__version__ = 2


class Hololens_Relay(Plugin):
    """Hololens_Relay
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
        self.menu.label = 'Hololens Relay'
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
            except Exception:
                ip = 'Your external ip'

        else:
            def set_address(new_address):
                if new_address.count(":") != 1:
                    logger.error("address format not correct")
                    return
                self.start_server(new_address)
                self.update_menu()

        help_str = 'The Hololens Relay is the bridge between Pupil Capture and the Hololens client. It uses UDP sockets to relay data.'
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Switch('use_primary_interface', self, setter=set_iface,
                                   label="Use primary network interface"))
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
        ipc_sub = zmq_tools.Msg_Receiver(context, self.g_pool.ipc_sub_url,
                                         ('gaze', 'notify.calibration.failed',
                                          'notify.calibration.successful'))
        poller.register(pipe, zmq.POLLIN)
        poller.register(ipc_sub.socket, zmq.POLLIN)
        remote_socket = None
        self.gaze_receiver = None
        self.calib_result_receiver = None

        while True:
            items = [sock for sock, _ in poller.poll()]
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

            if remote_socket.fileno() in items:
                self.on_recv(remote_socket, ipc_pub)

            if ipc_sub.socket in items:
                while ipc_sub.new_data:
                    # we should only receive gaze data
                    topic, payload = ipc_sub.recv()

                    # gaze events
                    if (self.gaze_receiver is not None and
                            remote_socket is not None and
                            topic.startswith('gaze')):
                        _, method, eye = payload['topic'].split('.')
                        if method == '2d':
                            data = b'EG%s%s%s' % (method[:1].encode(),
                                                  eye.encode(),
                                                  struct.pack('ff', *payload['norm_pos']))
                        elif method == '3d':
                            data = b'EG%s%s%s' % (method[:1].encode(),
                                                  eye.encode(),
                                                  struct.pack('fff', *payload['gaze_point_3d']))
                        else:
                            logger.error('Error while relaying gaze: "{}": {}'.format(topic, payload))
                            data = b'EGFError while relaying gaze'
                        remote_socket.sendto(data, self.gaze_receiver)

                    # calibration events
                    elif (self.calib_result_receiver is not None and
                            remote_socket is not None and
                            topic.startswith('notify.calibration.')):

                        if payload['subject'] == 'calibration.successful':
                            # event calibration successful
                            remote_socket.sendto(b'ECS', self.calib_result_receiver)
                            self.calib_result_receiver = None
                        elif payload['subject'] == 'calibration.failed':
                            # event calibration failed
                            remote_socket.sendto(b'ECF', self.calib_result_receiver)
                            self.calib_result_receiver = None

        remote_socket.close()
        self.thread_pipe = None

    def on_recv(self, socket, ipc_pub):
        try:
            byte_msg, sender = socket.recvfrom(2048)
        except OSError:
            return

        if byte_msg[:1] == b'R':  # reference point
            # read amount of msgpack bytes
            try:
                # relay reference point
                ipc_pub.socket.send_string('notify.calibration.add_ref_data', flags=zmq.SNDMORE)
                ipc_pub.socket.send(byte_msg[1:])
            except Exception as e:
                # respond with <error byte><headercode byte>[<byte>, ...]
                response = b'FRReference point mal-formatted or missing: %s' % str(e).encode()
            else:
                response = b'0R'
        elif byte_msg[:1] == b'S':
            self.gaze_receiver = sender
            response = b'0S'

        elif byte_msg[:1] == b's':
            self.gaze_receiver = None
            response = b'0s'

        elif byte_msg[:1] == b'I':
            mode = byte_msg[1:2]

            init_2d = mode == b'2'
            if init_2d:
                detection_mode = '2d'
                calib_method = 'HMD_Calibration'
            else:
                detection_mode = '3d'
                calib_method = 'HMD_Calibration_3D'

            ipc_pub.notify({'subject': 'start_plugin', 'name': calib_method})
            ipc_pub.notify({'subject': 'set_detection_mapping_mode',
                            'mode': detection_mode})
            ipc_pub.notify({'subject': 'eye_process.should_start.{}'.format(0),
                            'eye_id': 0, 'delay': .4})
            ipc_pub.notify({'subject': 'eye_process.should_start.{}'.format(1),
                            'eye_id': 1, 'delay': .2})
            response = b'0I'

        elif byte_msg[:1] == b'i':
            ipc_pub.notify({'subject': 'eye_process.should_stop', 'eye_id': 0})
            ipc_pub.notify({'subject': 'eye_process.should_stop', 'eye_id': 1})
            response = b'0i'

        elif byte_msg[:1] == b'C':
            width, height, outlier_threshold = struct.unpack('HHf', byte_msg[1:])
            ipc_pub.notify({'subject': 'calibration.should_start',
                            'hmd_video_frame_size': (width, height),
                            'outlier_threshold': outlier_threshold,
                            'translation_eye0': [-15., 0., 0.],
                            'translation_eye1': [15., 0., 0.]})
            response = b'0C'

        elif byte_msg[:1] == b'c':
            self.calib_result_receiver = sender
            ipc_pub.notify({'subject': 'calibration.should_stop'})
            response = b'0c'

        elif byte_msg[:1] == b'T':
            try:
                target = struct.unpack('f', byte_msg[1:])[0]
            except IndexError:
                response = b'FTTimestamp required'
            except Exception:
                response = b"FT'%s' cannot be converted to float." % (byte_msg[1])
            else:
                raw_time = self.g_pool.get_now()
                self.g_pool.timebase.value = raw_time-target
                response = b'0T'
        elif byte_msg[:1] == b'V':
            response = b'0V%s' % bytes(__version__)

        else:
            response = b'FFUnknown command. "%s"' % byte_msg

        if response[:1] == b'F':
            logger.error((b'Failed "%s": %s' % (response[1:2], response[2:])).decode())
        socket.sendto(response, sender)

    def get_init_dict(self):
        return {'port': self.port, 'host': self.host, 'use_primary_interface': self.use_primary_interface}

    def cleanup(self):
        """gets called when the plugin get terminated.
           This happens either voluntarily or forced.
        """
        self.stop_server()
