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

import tasklib.background

logger = logging.getLogger(__name__)


class PluginTaskManager:
    """
    Manager of tasks that makes it simpler and easier to use workers in Plugins.
    Parts of your plugin don't need to manage tasks themselves but get
    called when something interesting happens.

    How to use:
    1) Make your plugin an Observable.
    2) Create a PluginTaskManager() in your plugin.
    3) Give the manager e.g. to controllers and other parts of your plugin that need
       to start tasks.
    4) Call create_background_task() to add tasks, or add you own task that inherits
       from TaskInterface via add_task().
    """

    def __init__(self, plugin):
        self._tasks = []
        plugin.add_observer("recent_events", self.on_recent_events)
        plugin.add_observer("cleanup", self.on_cleanup)

    def __del__(self):
        self.on_cleanup()

    def create_background_task(
        self,
        name,
        routine_or_generator_function,
        pass_shared_memory=False,
        args=None,
        kwargs=None,
        patches=None,
    ):
        """
        Creates a managed background task.

        The task is started automatically during the next recent_events.
        If you want to start it immediately, you can call start() on the returned task.

        Right after you created the task, you can call add_observer on the returned
        task to get notified when your task completes, yields results etc.
        See tasklib.interface.py for a list of all events you can add observers to.

        Args:
            name (String): Name of the process for the task, only for identification
                purposes. Multiple processes might have the same name.
            routine_or_generator_function (Callable): Your task. Supports generator
                functions (= methods and functions that yield results) and routines
                (= methods and functions without yield).
            args: Arguments for the task.
            kwargs: Keyword arguments for the task.
            pass_shared_memory: If True, a shared memory object will be passed to the
                task (see tasklib.background.shared_memory.py). For this to work,
                your task needs to accept a keyword argument `shared_memory`.
            patches (collection of Patch): Patches that will be applied to your
                background process right after start. Use them if you need to fix
                something in the environment of the new process (see
                tasklib.background.patches.py).
                Per default, the IPC logging is patched.

        Returns:
            A new task with base class TaskInterface.

        """
        task = tasklib.background.create(
            name,
            routine_or_generator_function,
            pass_shared_memory,
            args,
            kwargs,
            patches,
        )
        self._tasks.append(task)
        return task

    def add_task(self, task):
        """
        Use this to add a custom task to the manager. Custom means that you inherit
        from TaskInterface to implement you own type of task.

        Similarly to create_task(), you don't need to start the task before adding
        it, but you can if you want.
        """
        self._tasks.append(task)

    def on_recent_events(self, _):
        for task in self._tasks.copy():
            if not task.started:
                task.start()
            if task.running:
                task.update()
            if task.ended:
                self._tasks.remove(task)

    def on_cleanup(self):
        self._kill_all_running_tasks()
        self._tasks = []

    def _kill_all_running_tasks(self, grace_period_per_task=None):
        for task in self._tasks:
            if task.running:
                task.kill(grace_period=grace_period_per_task)


class UniqueTaskManager(PluginTaskManager):
    """TaskManager ensuring tasks are unique by identifier"""

    def add_task(self, task_new, identifier: str):
        UniqueTaskManager._patch_task(task_new, identifier)
        task_duplicated = self._get_duplicated_task(identifier)
        if task_duplicated is not None:
            state = "running" if task_duplicated.running else "queued"
            logger.debug(f"Replacing {state} task with ID '{identifier}'")
            if task_duplicated.running:
                task_duplicated.kill(grace_period=None)
            self._tasks.remove(task_duplicated)
        super().add_task(task_new)

    def _get_duplicated_task(self, identifier):
        for task_prev in self._tasks:
            if task_prev._unique_task_identifier == identifier:
                return task_prev

    @staticmethod
    def _patch_task(task, identifier: str):
        task._unique_task_identifier = identifier
