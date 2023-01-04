"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import socket
from time import sleep

import os_utils
import zmq
import zmq_tools
from observable import Observable
from pyre import zhelper

os_utils.patch_pyre_zhelper_cdll()
logger = logging.getLogger(__name__)


class PupilRemoteController(Observable):

    """Pupil Remote Controller

    Send simple string messages to control Pupil Capture functions:
        'R' start recording with auto generated session name
        'R' rec_name' start recording and name new session name: rec_name
        'r' stop recording
        'C' start currently selected calibration
        'c' stop currently selected calibration
        'T 1234.56' Timesync: make timestamps count form 1234.56 from now on.
        't' get pupil capture timestamp returns a float as string.
        'v' get pupil software version string

        # IPC Backbone communication
        'PUB_PORT' return the current pub port of the IPC Backbone
        'SUB_PORT' return the current sub port of the IPC Backbone

    Mulitpart messages will be forwarded to the Pupil IPC Backbone.For high-frequency
        messages, it is recommended to use a PUSH socket instead.

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

    def on_pupil_remote_server_did_start(self, address: str):
        logger.debug(f"on_pupil_remote_server_did_start({address})")

    def on_pupil_remote_server_did_stop(self):
        logger.debug(f"on_pupil_remote_server_did_stop")

    def __init__(self, g_pool, host="*", use_primary_interface=True, **kwargs):
        assert type(host) == str
        port = int(g_pool.preferred_remote_port)

        # Global state
        self.g_pool = g_pool

        # Private state
        self.__primary_port = port
        self.__custom_port = port
        self.__custom_host = host
        self.__thread_pipe = None
        self.__use_primary_interface = use_primary_interface

        # Start the server on init
        self.__start_server(host=host, port=port)

    def get_init_dict(self):
        return {
            "host": self.__custom_host,
            "use_primary_interface": self.__use_primary_interface,
        }

    @property
    def primary_port(self) -> int:
        return self.__primary_port

    @primary_port.setter
    def primary_port(self, port: int):
        self.__primary_port = port
        self._restart_with_primary_interface()

    @property
    def use_primary_interface(self) -> bool:
        return self.__use_primary_interface

    @use_primary_interface.setter
    def use_primary_interface(self, value: bool):
        if self.__use_primary_interface == value:
            return  # No change
        self.__use_primary_interface = value
        if self.__use_primary_interface:
            self._restart_with_primary_interface()
        else:
            self._restart_with_custom_interface()

    @property
    def local_address(self) -> str:
        return f"127.0.0.1:{self.__primary_port}"

    @property
    def remote_address(self) -> str:
        try:
            external_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            external_ip = "Your external ip"
        return f"{external_ip}:{self.__primary_port}"

    @property
    def custom_address(self) -> str:
        return f"{self.__custom_host}:{self.__custom_port}"

    @custom_address.setter
    def custom_address(self, address: str):
        if address.count(":") != 1:
            logger.error("address format not correct")
            return
        self.__custom_host = address.split(":")[0]
        self.__custom_port = int(address.split(":")[1])
        self._restart_with_custom_interface()

    def restart_server(self, host: str, port: int):
        self.__stop_server()
        self.__start_server(host=host, port=port)

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        """
        self.__stop_server()

    ### PRIVATE

    def _restart_with_primary_interface(self):
        assert self.use_primary_interface  # sanity check
        self.restart_server(host="*", port=self.__primary_port)

    def _restart_with_custom_interface(self):
        assert not self.use_primary_interface  # sanity check
        self.restart_server(host=self.__custom_host, port=self.__custom_port)

    def __start_server(self, host: str, port: int):
        if self.__thread_pipe is not None:
            logger.warning("Pupil remote server already started")
            return

        new_address = f"{host}:{port}"
        self.__thread_pipe = zhelper.zthread_fork(
            self.g_pool.zmq_ctx, self.__thread_loop
        )
        self.__thread_pipe.send_string("Bind", flags=zmq.SNDMORE)
        self.__thread_pipe.send_string(f"tcp://{new_address}")
        response = self.__thread_pipe.recv_string()
        msg = self.__thread_pipe.recv_string()
        if response == "Bind OK":
            # TODO: Do we need to verify msg == new_address?
            self.on_pupil_remote_server_did_start(address=new_address)
            return

        # fail logic
        logger.error(msg)

        # for service we shut down
        if self.g_pool.app == "service":
            logger.error("Port already in use.")
            # NOTE: We don't want a should_stop notification, but a hard termination at
            # this point, because Service is still initializing at this point. This way
            # we can prevent the eye processes from starting, where otherwise we would
            # have to wait for them to be started until we can close them.
            self.g_pool.service_should_run = False
            return

        # for capture we try to bind to a arbitrary port on the first external interface
        else:
            self.__thread_pipe.send_string("Bind", flags=zmq.SNDMORE)
            self.__thread_pipe.send_string("tcp://*:*")
            response = self.__thread_pipe.recv_string()
            msg = self.__thread_pipe.recv_string()
            if response == "Bind OK":
                host, port = msg.split(":")
                self.host = host
                self.port = int(port)
            else:
                logger.error(msg)
                raise Exception("Could not bind to port")

    def __stop_server(self):
        if self.__thread_pipe is None:
            logger.warning("Pupil remote server already stopped")
            return
        self.__thread_pipe.send_string("Exit")
        while self.__thread_pipe:
            sleep(0.1)
        self.on_pupil_remote_server_did_stop()

    def __thread_loop(self, context, pipe):
        poller = zmq.Poller()
        ipc_pub = zmq_tools.Msg_Dispatcher(context, self.g_pool.ipc_push_url)
        poller.register(pipe, zmq.POLLIN)
        remote_socket = None

        while True:
            items = dict(poller.poll())
            if pipe in items:
                cmd = pipe.recv_string()
                if cmd == "Exit":
                    break
                elif cmd == "Bind":
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
                        pipe.send_string(
                            "Could not bind to Socket: {}. Reason: {}".format(
                                new_url, e
                            )
                        )
                    else:
                        pipe.send_string("Bind OK", flags=zmq.SNDMORE)
                        # `.last_endpoint` is already of type `bytes`
                        pipe.send(remote_socket.last_endpoint.replace(b"tcp://", b""))
                        poller.register(remote_socket, zmq.POLLIN)
            if remote_socket in items:
                self.__on_recv(remote_socket, ipc_pub)

        self.__thread_pipe = None

    def __on_recv(self, remote, ipc_pub):
        msg = remote.recv_string()
        if remote.get(zmq.RCVMORE):
            ipc_pub.socket.send_string(msg, flags=zmq.SNDMORE)
            while True:
                frame = remote.recv(flags=zmq.NOBLOCK)
                more_frames_coming = remote.get(zmq.RCVMORE)
                if more_frames_coming:
                    ipc_pub.socket.send(frame, flags=zmq.SNDMORE)
                else:
                    ipc_pub.socket.send(frame)
                    break
            response = "Message forwarded."
        elif msg == "SUB_PORT":
            response = self.g_pool.ipc_sub_url.split(":")[-1]
        elif msg == "PUB_PORT":
            response = self.g_pool.ipc_pub_url.split(":")[-1]
        elif msg[0] == "R":
            try:
                ipc_pub.notify(
                    {"subject": "recording.should_start", "session_name": msg[2:]}
                )
                response = "OK"
            except IndexError:
                response = "Recording command mal-formatted."
        elif msg[0] == "r":
            ipc_pub.notify({"subject": "recording.should_stop"})
            response = "OK"
        elif msg == "C":
            ipc_pub.notify({"subject": "calibration.should_start"})
            response = "OK"
        elif msg == "c":
            ipc_pub.notify({"subject": "calibration.should_stop"})
            response = "OK"
        elif msg[0] == "T":
            try:
                target = float(msg[2:])
            except Exception:
                response = f"'{msg[2:]}' cannot be converted to float."
            else:
                raw_time = self.g_pool.get_now()
                self.g_pool.timebase.value = raw_time - target
                response = "Timesync successful."
        elif msg[0] == "t":
            response = repr(self.g_pool.get_timestamp())
        elif msg[0] == "v":
            response = f"{self.g_pool.version}"
        else:
            response = "Unknown command."
        remote.send_string(response)
        logger.debug(f"Request: '{msg}', Response: '{response}'")
