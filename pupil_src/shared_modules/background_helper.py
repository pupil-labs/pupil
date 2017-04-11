'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from multiprocessing import Process, Pipe
from collections import namedtuple

TERM_SIGNAL = '$TERM'
Task_Proxy = namedtuple('Task_Proxy', ('process', 'cmd', 'data'))


def recent_events(pipe):
    # expects pipe to be of type multiprocessing.Connection
    try:
        # poll returns True when pipe was closed from the other side
        # Calling recv on such a closed pipe will raise an EOFError
        while pipe.poll(0):
            yield pipe.recv()
    except EOFError:
        pass


def all_events(pipe):
    # expects pipe to be of type multiprocessing.Connection
    try:
        while True:
            yield pipe.recv()
    except EOFError:
        pass


def start_background_task(task, args=(), kwargs={}, name='Pupil Background Helper'):
    cmd_pipe_to_bg, cmd_pipe_from_bg = Pipe()
    data_pipe_recv, data_pipe_send = Pipe(False)

    args_incl_pipes = (cmd_pipe_from_bg, data_pipe_send) + args
    proc = Process(target=task, name=name, args=args_incl_pipes, kwargs=kwargs)
    proc.start()

    return Task_Proxy(proc, cmd_pipe_to_bg, data_pipe_recv)


def cancel_background_task(proxy, return_remaining_msgs=False):
    # expects proxy to be of type Task_Proxy
    if proxy and proxy.process.is_alive():
        try:
            proxy.cmd.send(TERM_SIGNAL)
        except (OSError, BrokenPipeError):
            pass  # proxy.cmd was already closed
        if return_remaining_msgs:
            remaining_cmds = tuple(recent_events(proxy.cmd))
            remaining_data = tuple(recent_events(proxy.data))
            return remaining_cmds, remaining_data
