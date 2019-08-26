"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import logging
from queue import Queue
from queue import Empty as EmptyQueueException
import threading
import typing

from .base import EarlyCancellationError
from .base import Task_Proxy as Base_Task_Proxy


logger = logging.getLogger(__name__)


class Task_Proxy(Base_Task_Proxy):

    def __init__(self, name: str, generator, args=(), kwargs={}, **_):
        super().__init__(name=name, generator=generator, args=args, kwargs=kwargs)

        self._should_terminate_flag = False
        self._completed = False
        self._canceled = False

        pipe_recv, pipe_send = _Threading_Pipe(False)
        wrapper_args = self._prepare_wrapper_args(
            pipe_send, generator
        )
        wrapper_args.extend(args)
        self.process = threading.Thread(
            target=self._wrapper, name=name, args=wrapper_args, kwargs=kwargs
        )
        self.process.start()
        self.pipe = pipe_recv

    def fetch(self) -> typing.Iterator[typing.Any]:
        if self.completed or self.canceled:
            return

        while self.pipe.poll(0):
            try:
                datum = self.pipe.recv()
            except EOFError:
                logger.debug("Process canceled be user.")
                self._canceled = True
                return
            else:
                if isinstance(datum, StopIteration):
                    self._completed = True
                    return
                elif isinstance(datum, EarlyCancellationError):
                    self._canceled = True
                    return
                elif isinstance(datum, Exception):
                    raise datum
                else:
                    yield datum

    def cancel(self, timeout=1):
        if not (self.completed or self.canceled):
            self._should_terminate_flag = True
            for x in self.fetch():
                # fetch to flush pipe to allow process to react to cancel comand.
                pass
        if self.process is not None:
            self.process.join(timeout)
            assert not self.process.is_alive()
            self.process = None

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def canceled(self) -> bool:
        return self._canceled

    def _prepare_wrapper_args(self, *args):
        return list(args)

    def _wrapper(self, pipe, generator, *args, **kwargs):
        try:
            logger.debug("Entering _wrapper")

            for datum in generator(*args, **kwargs):
                if self._should_terminate_flag:
                    raise EarlyCancellationError("Task was cancelled")
                pipe.send(datum)
        except Exception as e:
            pipe.send(e)
            if not isinstance(e, EarlyCancellationError):
                import traceback

                logger.info(traceback.format_exc())
        else:
            pipe.send(StopIteration())
        finally:
            pipe.close()
            logger.debug("Exiting _wrapper")


def _Threading_Pipe(duplex: bool):
    shared_queue = Queue()
    pipe_recv = _Threading_Connection(send_queue=shared_queue, recv_queue=shared_queue)
    pipe_send = pipe_recv
    return pipe_recv, pipe_send


class _Threading_Connection:
    def __init__(self, send_queue: Queue, recv_queue: Queue):
        self._send_queue = send_queue
        self._recv_queue = recv_queue
        self._recv_buffer = collections.deque(maxlen=1)

    # send

    def send(self, item):
        self._send_queue.put_nowait(item)

    def close(self):
        while self._send_queue.unfinished_tasks > 0:
            self._send_queue.task_done()
        self._send_queue.join()

    # recv

    def poll(self, timeout=None) -> bool:
        if len(self._recv_buffer) > 0:
            return True

        def is_timeout_valid() -> bool:
            return timeout is not None and timeout > 0

        try:
            value = self._recv_queue.get(block=is_timeout_valid(), timeout=timeout)
        except EmptyQueueException:
            return False

        self._recv_buffer.append(value)
        return True

    def recv(self) -> typing.Any:
        if len(self._recv_buffer) > 0:
            return self._recv_buffer.pop()
        try:
            value = self._recv_queue.get_nowait()
        except EmptyQueueException:
            raise

        return value
