"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

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
    # this needs to be set once by the foreground process before the patch can be used
    ipc_push_url = None

    def apply(self):
        """
        ZMQ_handler sockets from the foreground thread are broken in the background.
        Solution: Remove all potential broken handlers and replace by new ones.

        Caveat: If a broken handler is present it is inconsistent across environments.
        """
        assert self.ipc_push_url, "`ipc_push_url` was not set by foreground process"
        import logging
        import zmq
        import zmq_tools

        root_logger = logging.getLogger(name=None)
        del root_logger.handlers[:]
        zmq_ctx = zmq.Context()
        handler = zmq_tools.ZMQ_handler(zmq_ctx, self.ipc_push_url)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.NOTSET)
