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
import multiprocessing as mp
import typing
from collections import namedtuple

from tasklib.background.patches import Patch
from tasklib.background.shared_memory import SharedMemory
from tasklib.interface import TaskInterface

_TaskYieldSignal = namedtuple("_TaskYieldSignal", "datum")
_TaskCompletedSignal = namedtuple("_TaskCompletedSignal", "return_value")
_TaskCanceledSignal = namedtuple("_TaskCanceledSignal", "")
_TaskExceptionSignal = namedtuple("_TaskExceptionSignal", ["exception", "traceback"])


class BackgroundTask(TaskInterface, metaclass=abc.ABCMeta):
    def __init__(
        self, name, generator_function, pass_shared_memory, args, kwargs, patches
    ):
        super().__init__()

        self._shared_memory = SharedMemory()
        if pass_shared_memory:
            kwargs["shared_memory"] = self._shared_memory

        pipe_recv, pipe_send = mp.Pipe(duplex=False)
        self.process = self.get_process(
            name, generator_function, args, kwargs, pipe_send, patches
        )
        self.process.daemon = True
        self.pipe_recv = pipe_recv

    @abc.abstractmethod
    def get_process(self, name, generator_function, args, kwargs, pipe_send):
        pass

    @property
    def progress(self):
        return self._shared_memory.progress

    def start(self):
        super().start()
        self.process.start()

    def cancel_gracefully(self):
        super().cancel_gracefully()
        self._ask_process_to_shut_down()

    def kill(self, grace_period):
        super().kill(grace_period)
        self._ask_process_to_shut_down()
        if grace_period:
            # attention: we could pass None to join, but there it means
            # "no timeout", which is the exact opposite of what we want in this case
            # (don't wait at all)!
            self.process.join(grace_period)
        if self.process.is_alive():
            self.process.terminate()
        self.on_canceled_or_killed()

    def _ask_process_to_shut_down(self):
        self._shared_memory.should_terminate_flag = True

    def update(self):
        super().update()

        while self.pipe_recv.poll(timeout=0):
            signal = self.pipe_recv.recv()
            if self._shared_memory.should_terminate_flag:
                should_continue = self._handle_signal_if_canceled(signal)
            else:
                should_continue = self._handle_signal_normally(signal)
            if not should_continue:
                return

    def _handle_signal_normally(self, signal):
        if isinstance(signal, _TaskCompletedSignal):
            self.on_completed(signal.return_value)
            return False
        elif isinstance(signal, _TaskExceptionSignal):
            # Unfortunately, background exceptions raised in the foreground don't
            # have a proper traceback.
            # If you are debugging an exception, you can print datum.traceback to
            # get the traceback in the other process. Just uncomment:
            # print(signal.traceback)
            # If this happens often, we can consider using tblib to send tracebacks
            # to the foreground (see https://stackoverflow.com/a/26096355)
            self.on_exception(signal.exception)
            return False
        elif isinstance(signal, _TaskYieldSignal):
            self.on_yield(signal.datum)
            return True
        else:
            raise ValueError(
                "Received unknown signal {} from background " "process".format(signal)
            )

    def _handle_signal_if_canceled(self, signal):
        if isinstance(
            signal, (_TaskCanceledSignal, _TaskCompletedSignal, _TaskExceptionSignal)
        ):
            self.on_canceled_or_killed()
            return False
        elif isinstance(signal, _TaskYieldSignal):
            return True
        else:
            raise ValueError(
                "Received unknown signal {} from background " "process".format(signal)
            )


class BackgroundGeneratorFunction(BackgroundTask):
    def get_process(self, name, generator_function, args, kwargs, pipe_send, patches):
        wrapper_kwargs = {
            "pipe_send": pipe_send,
            # passing the shared_memory to generator_function is optional (see parameter
            # pass_shared_memory above), but we always pass it to the wrapper. The
            # wrapper can terminate between yields even if generator_function does
            # not support graceful cancellation.
            "shared_memory": self._shared_memory,
            "generator_function": generator_function,
            "args": args,
            "kwargs": kwargs,
            "patches": patches,
        }

        return mp.Process(target=_generator_wrapper, name=name, kwargs=wrapper_kwargs)


def _generator_wrapper(
    pipe_send, generator_function, args, kwargs, patches, shared_memory
):
    """Executed in background, pipes results to foreground"""
    try:
        for patch in patches:
            patch.apply()
        for datum in generator_function(*args, **kwargs):
            if shared_memory.should_terminate_flag:
                pipe_send.send(_TaskCanceledSignal())
                return  # will also trigger "finally"
            pipe_send.send(_TaskYieldSignal(datum))
    except Exception as e:
        import traceback

        from rich import print

        print(traceback.format_exc())
        pipe_send.send(_TaskExceptionSignal(e, traceback.format_exc()))
    else:
        pipe_send.send(_TaskCompletedSignal(return_value=None))
    finally:
        pipe_send.close()


class BackgroundRoutine(BackgroundTask):
    def get_process(self, name, routine, args, kwargs, pipe_send, patches):
        wrapper_kwargs = {
            "pipe_send": pipe_send,
            "routine": routine,
            "args": args,
            "kwargs": kwargs,
            "patches": patches,
        }

        return mp.Process(target=_routine_wrapper, name=name, kwargs=wrapper_kwargs)


def _routine_wrapper(pipe_send, routine, args, kwargs, patches):
    try:
        for patch in patches:
            patch.apply()
        return_value = routine(*args, **kwargs)
        pipe_send.send(_TaskCompletedSignal(return_value))
    except Exception as e:
        import traceback

        pipe_send.send(_TaskExceptionSignal(e, traceback.format_exc()))
    finally:
        pipe_send.close()


GFY = typing.TypeVar("GFY")  # Generator function yield type
GFS = typing.TypeVar("GFS")  # Generator function send type
GFR = typing.TypeVar("GFR")  # Generator function return type

On_Started_Observer = typing.Callable[[], None]
On_Yield_Observer = typing.Callable[[GFY], None]
On_Completed_Observer = typing.Callable[[GFR], None]
On_Ended = typing.Callable[[], None]
On_Exception = typing.Callable[[Exception], None]
On_Canceled_Or_Killed = typing.Callable[[], None]


class TypedBackgroundGeneratorFunction(
    BackgroundGeneratorFunction, typing.Generic[GFY, GFS, GFR]
):
    def __init__(
        self,
        name: str,
        generator_function: typing.Callable[..., typing.Generator[GFY, GFS, GFR]],
        args: typing.List[typing.Any] = [],
        kwargs: typing.Mapping[str, typing.Any] = {},
        pass_shared_memory: bool = False,
        patches: typing.Iterable[typing.Type[Patch]] = tuple(),
    ):
        super().__init__(
            name=name,
            generator_function=generator_function,
            pass_shared_memory=pass_shared_memory,
            args=args,
            kwargs=kwargs,
            patches=patches,
        )

    def add_observers(
        self,
        on_started: typing.Optional[On_Started_Observer] = None,
        on_yield: typing.Optional[On_Yield_Observer] = None,
        on_completed: typing.Optional[On_Completed_Observer] = None,
        on_ended: typing.Optional[On_Ended] = None,
        on_exception: typing.Optional[On_Exception] = None,
        on_canceled_or_killed: typing.Optional[On_Canceled_Or_Killed] = None,
    ):
        if on_started:
            self.add_observer("on_started", on_started)
        if on_yield:
            self.add_observer("on_yield", on_yield)
        if on_completed:
            self.add_observer("on_completed", on_completed)
        if on_ended:
            self.add_observer("on_ended", on_ended)
        if on_exception:
            self.add_observer("on_exception", on_exception)
        if on_canceled_or_killed:
            self.add_observer("on_canceled_or_killed", on_canceled_or_killed)
