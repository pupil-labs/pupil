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

from observable import Observable


class TaskInterface(Observable, metaclass=abc.ABCMeta):
    def __init__(self):
        self._started = False
        self._completed = False
        self._canceled_or_killed = False

    def on_started(self):
        pass

    def on_yield(self, yield_value):
        """
        Add an observer to this to get notified every time a background task yields a
        new result.

        This is only called if the task is a generator function.
        """
        pass

    def on_completed(self, return_value_or_none):
        """
        Add an observer to this to get notified when a task completed successfully.
        This is not called if the task was canceled, raised an exception etc.

        Args:
            return_value_or_none: Return value of the background task or None if it
                did not return a value. For generator functions, this is always None.
        """
        self._completed = True
        self.on_ended()

    def on_ended(self):
        """
        Add an observer to this to get notified when a task finished IN ANY WAY.
        This is called even if the task was canceled, raised an exception, etc.
        """
        pass

    def on_exception(self, exception):
        """
        Add an observer to this to get notified when a task terminates because it
        raised an exception.
        """
        self._canceled_or_killed = True
        self.on_ended()

    def on_canceled_or_killed(self):
        """
        Add an observer to this to get notified when a task was terminated by
        cancel_gracefully() or kill().
        """
        self._canceled_or_killed = True
        self.on_ended()

    @property
    def started(self):
        """
        True if the task was ever started. Still True after it ended!
        """
        return self._started

    @property
    def running(self):
        return self.started and not self.ended

    @property
    @abc.abstractmethod
    def progress(self):
        """
        Progress as reported by the task. Usually from 0.0 to 1.0.
        If the task does not report its progress, this will always be 0.0.
        """
        pass

    @property
    def completed(self):
        """
        True if the task completed successfully. It was not canceled and did not
        raise exceptions.
        """
        return self._completed

    @property
    def ended(self):
        """
        True if the task ended in any way. Either successfully, or it was canceled,
        or it raised an exception.
        """
        return self.completed or self.canceled_or_killed

    @property
    def canceled_or_killed(self):
        """
        True if the task was terminated from the foreground.
        """
        return self._canceled_or_killed

    @abc.abstractmethod
    def start(self):
        """
        Start the task.

        Make sure to add any observers before calling this.

        Raises:
            ValueError: If the task was already started.
        """
        if self.started:
            raise ValueError("Task already started!")
        self._started = True
        self.on_started()

    @abc.abstractmethod
    def cancel_gracefully(self):
        """
        Ask the background process to shut itself down at the next occasion.

        For generator functions, this will be after the next yield.
        Routines cannot be interrupted, so they will just terminate normally.

        After this was called, any results by the process are ignored. You will not
        get notified about any yields, return values, or exceptions.
        However, on_canceled_or_killed() will be triggered when the process terminates.

        Compared to kill():
        - The task is not terminated immediately and there are no guarantees that it
          gets terminated at all. However, all processes are started as daemons,
          so at least they terminate when the main process terminates.
        - The foreground process does not block.
        - The background can clean up normally and will not break shared resources etc.
        - on_canceled_or_killed() will be called in the next update() after the task
          terminated.

        Raises:
            ValueError: If the task is not running, i.e. it was never started or it
                already ended.
        """
        if not self.running:
            raise ValueError("Task not running, can only cancel running tasks!")

    @abc.abstractmethod
    def kill(self, grace_period):
        """
        Try to cancel the task gracefully and kill it if that's not successful.

        Compared to cancel_gracefully():
        - You can be sure that the task gets terminated during the grace
          period or immediately after.
        - The foreground process blocks until the process terminates or the grace
          period is over.
        - You likely break resources shared between your background process and the
          foreground.
        - You can be sure that on_canceled_or_killed() gets called immediately and not
          during the next update().

        Args:
            grace_period (Number): Seconds to wait for the process to shut itself
                down. If None, there is no grace period.

        Raises:
            ValueError: If the task is not running, i.e. it was never started or it
                already ended.
        """
        if not self.running:
            raise ValueError("Task not running, can only kill running tasks!")

    @abc.abstractmethod
    def update(self):
        """
        Fetches results and exceptions from the background process and calls the
        corresponding events to trigger observers.

        You should not call this when using a PluginTaskManager.

        Raises:
            ValueError: If the task is not running, i.e. it was never started or it
                already ended.
        """
        if not self.running:
            raise ValueError("Task not running, can only update running tasks!")
