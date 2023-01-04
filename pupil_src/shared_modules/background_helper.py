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
import multiprocessing as mp
import signal
from ctypes import c_bool

import zmq
import zmq_tools

logger = logging.getLogger(__name__)


class EarlyCancellationError(Exception):
    pass


class Task_Proxy:
    """Future like object that runs a given generator in the background and returns is able to return the results incrementally"""

    def __init__(self, name, generator, args=(), kwargs={}, context=...):
        super().__init__()
        if context is ...:
            context = mp.get_context()

        self._should_terminate_flag = context.Value(c_bool, 0)
        self._completed = False
        self._canceled = False

        pipe_recv, pipe_send = context.Pipe(False)
        wrapper_args = self._prepare_wrapper_args(
            pipe_send, self._should_terminate_flag, generator
        )
        wrapper_args.extend(args)
        self.process = context.Process(
            target=self._wrapper, name=name, args=wrapper_args, kwargs=kwargs
        )
        self.process.daemon = True
        self.process.start()
        self.pipe = pipe_recv

    def _wrapper(self, pipe, _should_terminate_flag, generator, *args, **kwargs):
        """Executed in background, pipes generator results to foreground

        All exceptions are caught, forwarded to the foreground, and raised in
        `Task_Proxy.fetch()`. This allows users to handle failure gracefully
        as well as raising their own exceptions in the background task.
        """

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            # NOTE: Interrupt is handled in world/service/player which are responsible
            # for shutting down the background process properly

        signal.signal(signal.SIGINT, interrupt_handler)
        try:
            self._change_logging_behavior()
            logger.debug("Entering _wrapper")

            for datum in generator(*args, **kwargs):
                if _should_terminate_flag.value:
                    raise EarlyCancellationError("Task was cancelled")
                pipe.send(datum)
            pipe.send(StopIteration())
        except BrokenPipeError:
            # process canceled from outside
            pass
        except Exception as e:
            try:
                pipe.send(e)
            except BrokenPipeError:
                # process canceled from outside
                pass
            if not isinstance(e, EarlyCancellationError):
                import traceback

                logger.info(traceback.format_exc())
        finally:
            pipe.close()
            logger.debug("Exiting _wrapper")

    def _prepare_wrapper_args(self, *args):
        return list(args)

    def _change_logging_behavior(self):
        pass

    def fetch(self):
        """Fetches progress and available results from background"""
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
            self._should_terminate_flag.value = True
            for x in self.fetch():
                # fetch to flush pipe to allow process to react to cancel comand.
                pass
        if self.process is not None:
            self.process.join(timeout)
            self.process = None

    @property
    def completed(self):
        return self._completed

    @property
    def canceled(self):
        return self._canceled


class IPC_Logging_Task_Proxy(Task_Proxy):
    push_url = None

    def _prepare_wrapper_args(self, *args):
        return [*args, self.push_url]

    def _wrapper(
        self, pipe, _should_terminate_flag, generator, push_url, *args, **kwargs
    ):
        self.push_url = push_url
        super()._wrapper(pipe, _should_terminate_flag, generator, *args, **kwargs)

    def _change_logging_behavior(self):
        """
        ZMQ_handler sockets from the foreground thread are broken in the background.
        Solution: Remove all potential broken handlers and replace by new oneself.

        Caveat: If a broken handler is present it is incosistent across environments.
        """
        assert self.push_url, "`push_url` was not set by foreground process"
        del logger.root.handlers[:]
        zmq_ctx = zmq.Context()
        handler = zmq_tools.ZMQ_handler(zmq_ctx, self.push_url)
        logger.root.addHandler(handler)
        logger.root.setLevel(logging.NOTSET)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(processName)s - [%(levelname)s] %(name)s: %(message)s",
    )

    def example_generator(mu=0.0, sigma=1.0, steps=100):
        r"""samples `N(\mu, \sigma^2)`"""
        from time import sleep

        import numpy as np

        for i in range(steps):
            # yield progress, datum
            yield (i + 1) / steps, sigma * np.random.randn() + mu
            sleep(np.random.rand() * 0.1)

    # initialize task proxy
    task = Task_Proxy(
        "Background", example_generator, args=(5.0, 3.0), kwargs={"steps": 100}
    )

    from time import sleep, time

    start = time()
    maximal_duration = 2.0
    while time() - start < maximal_duration:
        # fetch all available results
        for progress, random_number in task.fetch():
            logger.debug(f"[{progress * 100:3.0f}%] {random_number:0.2f}")

        # test if task is completed
        if task.completed:
            break
        sleep(1.0)

    logger.debug("Canceling task")
    task.cancel(timeout=1)
    logger.debug("Task done")
