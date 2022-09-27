from dataclasses import dataclass
import multiprocessing


import logging
import zmq_tools


def event_loop(*args, **kwargs):
    with SharedCameraProcessManager(*args, **kwargs) as scpm:
        pass


@dataclass
class SharedCameraProcessManager:
    timebase: multiprocessing.Value
    ipc_pub_url: str
    ipc_sub_url: str
    ipc_push_url: str
    user_dir: str
    debug: bool = False
    skip_driver_installation: bool = False

    def __enter__(self):
        self._setup_networking()
        self._setup_logging()

    def _setup_networking(self):
        import zmq

        zmq_ctx = zmq.Context()
        ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
        notify_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url, topics=("notify",))

    def _setup_logging(self):
        # log setup
        logging.getLogger("OpenGL").setLevel(logging.ERROR)
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
        # create logger for the context of this function
        logger = logging.getLogger(__name__)
