"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc


class Patch(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self):
        pass


class IPCLoggingPatch(Patch):
    """
    ZMQ_handler sockets from the foreground thread are broken in the background.
    Solution: Remove all potential broken handlers and replace by new ones.

    Caveat: If a broken handler is present it is inconsistent across environments.
    """

    # this needs to be set once by the foreground process before the patch can be used
    ipc_push_url = None

    def __init__(self):
        assert (
            IPCLoggingPatch.ipc_push_url
        ), "`ipc_push_url` was not set by foreground process"
        # copy because object attributes get copied to background processes,
        # but class attributes do not
        self.ipc_push_url_copy = IPCLoggingPatch.ipc_push_url

    def apply(self):
        import logging

        import zmq
        import zmq_tools

        root_logger = logging.getLogger(name=None)
        del root_logger.handlers[:]
        zmq_ctx = zmq.Context()
        handler = zmq_tools.ZMQ_handler(zmq_ctx, self.ipc_push_url_copy)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.NOTSET)


class KeyboardInterruptHandlerPatch(Patch):
    def apply(self):
        import signal

        signal.signal(signal.SIGINT, self._debug_log_interrupt_trace)

    def _debug_log_interrupt_trace(self, sig, frame):
        import logging
        import traceback

        logger = logging.getLogger(__name__)
        trace = traceback.format_stack(f=frame)
        logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
        # NOTE: Interrupt is handled in world/service/player which are responsible for
        # shutting down the task properly
