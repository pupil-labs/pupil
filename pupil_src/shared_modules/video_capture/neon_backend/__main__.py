import contextlib
import ctypes
import multiprocessing
import os
import pathlib
import sys
from typing import Tuple

from .background import BackgroundCameraSharingManager
from .definitions import NEON_SHARED_EYE_FRAME_TOPIC


def main():
    timebase = multiprocessing.Value(ctypes.c_double)
    timebase.value = 0.0

    ipc_pub_url, ipc_push_url, ipc_sub_url = ipc()

    manager = BackgroundCameraSharingManager(
        timebase=timebase,
        ipc_pub_url=ipc_pub_url,
        ipc_push_url=ipc_push_url,
        ipc_sub_url=ipc_sub_url,
        topic_prefix=NEON_SHARED_EYE_FRAME_TOPIC,
    )
    with contextlib.suppress(KeyboardInterrupt):
        manager._background_process.join()
    manager.stop()


def ipc() -> Tuple[str, str, str]:
    from threading import Thread

    import zmq

    zmq_ctx = zmq.Context()

    # Let the OS choose the IP and PORT
    ipc_pub_url = "tcp://*:*"
    ipc_sub_url = "tcp://*:*"
    ipc_push_url = "tcp://*:*"

    # Binding IPC Backbone Sockets to URLs.
    # They are used in the threads started below.
    # Using them in the main thread is not allowed.
    xsub_socket = zmq_ctx.socket(zmq.XSUB)
    xsub_socket.bind(ipc_pub_url)
    ipc_pub_url = xsub_socket.last_endpoint.decode("utf8").replace(
        "0.0.0.0", "127.0.0.1"
    )

    xpub_socket = zmq_ctx.socket(zmq.XPUB)
    xpub_socket.bind(ipc_sub_url)
    ipc_sub_url = xpub_socket.last_endpoint.decode("utf8").replace(
        "0.0.0.0", "127.0.0.1"
    )

    pull_socket = zmq_ctx.socket(zmq.PULL)
    pull_socket.bind(ipc_push_url)
    ipc_push_url = pull_socket.last_endpoint.decode("utf8").replace(
        "0.0.0.0", "127.0.0.1"
    )

    # Starting communication threads:
    # A ZMQ Proxy Device serves as our IPC Backbone
    ipc_backbone_thread = Thread(
        target=zmq.proxy, args=(xsub_socket, xpub_socket), daemon=True
    )
    ipc_backbone_thread.start()

    pull_pub = Thread(
        target=pull_pub_thread, args=(ipc_pub_url, pull_socket), daemon=True
    )
    pull_pub.start()

    log_thread = Thread(target=log_loop_thread, args=(ipc_sub_url, True), daemon=True)
    log_thread.start()

    delay_thread = Thread(
        target=delay_proxy_thread, args=(ipc_push_url, ipc_sub_url), daemon=True
    )
    delay_thread.start()

    del xsub_socket, xpub_socket, pull_socket

    return ipc_pub_url, ipc_push_url, ipc_sub_url


# Reliable msg dispatch to the IPC via push bridge.
def pull_pub_thread(ipc_pub_url, pull):
    import zmq

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.connect(ipc_pub_url)

    while True:
        m = pull.recv_multipart()
        pub.send_multipart(m)


# The delay proxy handles delayed notififications.
def delay_proxy_thread(ipc_pub_url, ipc_sub_url):
    import zmq
    import zmq_tools

    ctx = zmq.Context.instance()
    sub = zmq_tools.Msg_Receiver(ctx, ipc_sub_url, ("delayed_notify",))
    pub = zmq_tools.Msg_Dispatcher(ctx, ipc_pub_url)
    poller = zmq.Poller()
    poller.register(sub.socket, zmq.POLLIN)
    waiting_notifications = {}

    TOPIC_CUTOFF = len("delayed_")

    while True:
        if poller.poll(timeout=250):
            # Recv new delayed notification and store it.
            topic, n = sub.recv()
            n["__notify_time__"] = time() + n["delay"]
            waiting_notifications[n["subject"]] = n
        # When a notifications time has come, pop from dict and send it as notification
        for s, n in list(waiting_notifications.items()):
            if n["__notify_time__"] < time():
                n["topic"] = n["topic"][TOPIC_CUTOFF:]
                del n["__notify_time__"]
                del n["delay"]
                del waiting_notifications[s]
                pub.notify(n)


# Recv log records from other processes.
def log_loop_thread(ipc_sub_url, log_level_debug):
    import logging

    import zmq
    import zmq_tools
    from rich.logging import RichHandler

    # Get the root logger
    logger = logging.getLogger()
    # set log level
    logger.setLevel(logging.NOTSET)
    # Stream to file
    fh = logging.FileHandler("neon_backend.log", mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(processName)s - [%(levelname)s] %(name)s: %(message)s"
        )
    )
    logger.addHandler(fh)
    # Stream to console.

    ch = RichHandler(
        level=logging.DEBUG if log_level_debug else logging.INFO,
        rich_tracebacks=False,
    )
    ch.setFormatter(logging.Formatter("%(processName)s - %(name)s: %(message)s"))

    logger.addHandler(ch)
    # IPC setup to receive log messages. Use zmq_tools.ZMQ_handler to send messages to here.
    sub = zmq_tools.Msg_Receiver(zmq.Context(), ipc_sub_url, topics=("logging",))
    while True:
        topic, msg = sub.recv()
        record = logging.makeLogRecord(msg)
        logger.handle(record)


if __name__ == "__main__":
    shared_modules = pathlib.Path(__file__).parent.parent.parent
    print(f"{shared_modules=}")
    sys.path.append(str(shared_modules))
    main()
