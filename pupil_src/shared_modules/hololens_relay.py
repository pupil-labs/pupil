'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

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
    """UDP relay for Pupil to Hololens communication

    This plugins defines a very narrow api that is custimized to work with
    the Pupil-Hololens-integration. On startup a udp socket is opened that
    can be changed in the user settings.

    The Hololens side will be called the `client` and the plugin the `server`.

    Communication is initialized by the client. The server determines the
    address of the client by calling `recvfrom()`. All server respones are sent
    to the most recent client address. This means that the client needs to
    listen on the same socket that it uses to send requests.

    All messages start with an ascii-encoded header byte. Server acknowledgments
    start either with ascii `0` on success or `F` on failure, followed by the
    request's header byte. In some cases, there are further bytes containing
    response-specific information.

    Events are server-side initiated messages that are not responses to specific
    requests. Events include gaze data and calibration results. See below for
    details.

    - Messages are composed of blocks. The default type of a block is an unsigned
        ascii character. All blocks are little endian.
    - <type "name"> denotes a block named "name" with its value encoded as `type`.
        Depending on the type the block can take up more than one byte in space,
        e.g. <float32> requires 4 bytes.
    - (X|3|<type>) represents of a group of mutually exclusive blocks.
    - [<type>, ...] represents a list of zero or more blocks


    ### Request messages

    I(2|3)      Initializes the relay in 2d or 3d mode. This request triggers
                    1. Sets the detection and mapping mode to its corresponding value
                    2. Launches both eye processs
                    3. Starts the HMD Calibration (3D) plugin

    i           Deinit. Closes the eye processes but does not stop the gaze broadcast

    T<float32 "timestamp">
                Sets Pupil's timebase to "timestamp"

    C<uint16 "width"><uint16 "height"><float32 "outlier-threshold">
                Start calibration with "width"x"height" as hmd frame size
                and "outlier-threshold" as threshold for the 2d hmd calibration.

    c           Stop calibration

    S           Start gaze broadcast

    s           Stop gaze broadcast

    R[<bytes>, ...]
                List of reference points encoded with msgpack. This message
                should not exceed 2048 bytes. The Python socket.recvfrom()
                implemetation drops all exceeding bytes upon reading.

    V           Protocol version. This message is responded to with:
                    0V[<bytes>, ...]

    ### Responses

    0<byte "header">[<byte>, ...]
                Response to a successful "header" request

    F<byte "header">[<byte>, ...]
                Error response to "header" including an error message

    ### Events

    EC(S|F|U)   Indicates a *S*uccessful, *F*ailed, or *U*nknown calibration result.

    EG(2|3)(0|1|2)[<float32 "gaze component">]
                Gaze datum. The first group indicates the number of gaze components.
                The second group indicates if the datum belongs to a specific eye,
                0 or 1, or if it is a binocular result, 2.

    ### Example

    A typical sequence of events between client (C) and server (S) would
    look similar to this:

    C:      I2
    S:      0I

    C:      T<time>
    S:      0T

    C:      C<width><height><threshold>
    S:      0C

    C:      R<refpoint bytes>
    S:      0R
        .
        .
        .

    C:      c
    S:      0c
    S:      ECS

    C:      S
    S:      0S

    S:      EG20<gaze x><gaze y>
        .
        .
        .

    C:      s
    S:      0s

    C:      i
    S:      0i
    """
    icon_chr = chr(0xec21)
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
                        method, eye = payload['topic'].split('.')[1:3]
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
            logger.info('{}:{} subscribed'.format(*sender))
            response = b'0S'

        elif byte_msg[:1] == b's':
            self.gaze_receiver = None
            logger.info('{}:{} unsubscribed'.format(*sender))
            response = b'0s'

        elif byte_msg[:1] == b'I':
            logger.info('{}:{} connected'.format(*sender))
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
            logger.info('{}:{} disconnected'.format(*sender))
            ipc_pub.notify({'subject': 'eye_process.should_stop', 'eye_id': 0})
            ipc_pub.notify({'subject': 'eye_process.should_stop', 'eye_id': 1})
            response = b'0i'

        elif byte_msg[:1] == b'C':
            self.calib_result_receiver = sender
            width, height, outlier_threshold = struct.unpack('HHf', byte_msg[1:])
            ipc_pub.notify({'subject': 'calibration.should_start',
                            'hmd_video_frame_size': (width, height),
                            'outlier_threshold': outlier_threshold,
                            'translation_eye0': [27.84765, 0., 0.],
                            'translation_eye1': [-27.84765, 0., 0.]})
            response = b'0C'

        elif byte_msg[:1] == b'c':
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
