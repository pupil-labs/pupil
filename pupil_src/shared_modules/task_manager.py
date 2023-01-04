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
import logging

import background_helper as bh
from plugin import Plugin
from pyglui import ui

logger = logging.getLogger(__name__)


class TaskManager(Plugin, abc.ABC):
    """
    Base for plugins that need to perform 'tasks' (possibly running simultaneously).

    You can set the maximum number of simultaneous tasks and if you add more they will
    wait in a queue. Also all finished tasks are shown.

    Tasks can be canceled by the user (all or individual ones).

    To use this, you just have to add tasks via add_task() and override customize_menu()
    The latter should be used to add custom controls for your tasks
    """

    def __init__(self, g_pool, max_concurrent_tasks=2):
        super().__init__(g_pool)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.managed_tasks = []
        self.task_container = ui.Growing_Menu("Tasks")

    def add_task(self, managed_task):
        managed_task.ui.menu.collapsed = True
        managed_task.ui.label = "Queued"
        self.managed_tasks.append(managed_task)
        self.task_container.insert(0, managed_task.ui.menu)

    def cancel_all_tasks(self):
        for managed_task in self.managed_tasks:
            managed_task.cancel()
        self.managed_tasks = []
        del self.task_container[:]

    def remove_completed_tasks(self):
        completed_tasks = []
        for task in self.managed_tasks:
            if task.completed:
                completed_tasks.append(task)
        self._remove_tasks(completed_tasks)

    def init_ui(self):
        self.add_menu()
        self.customize_menu()
        self._add_manager_buttons_to_menu()
        self.menu.append(self.task_container)

    @abc.abstractmethod
    def customize_menu(self):
        pass

    def _add_manager_buttons_to_menu(self):
        self.menu.append(ui.Button("Cancel all", self.cancel_all_tasks))
        self.menu.append(
            ui.Button("Remove finished from list", self.remove_completed_tasks)
        )

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.cancel_all_tasks()

    def recent_events(self, events):
        self._manage_current_tasks()

    def _manage_current_tasks(self):
        self._remove_canceled_tasks()
        self._update_running_tasks()

    def _remove_canceled_tasks(self):
        remove_tasks = []
        for managed_task in self.managed_tasks:
            if managed_task.canceled:
                remove_tasks.append(managed_task)
        self._remove_tasks(remove_tasks)

    def _update_running_tasks(self):
        num_running_tasks = 0
        for managed_task in self.managed_tasks:
            if managed_task.queued:
                if num_running_tasks < self.max_concurrent_tasks:
                    managed_task.start()
                    managed_task.ui.label = "Running"
                    managed_task.ui.menu.collapsed = False
                    num_running_tasks += 1
            elif managed_task.completed:
                if managed_task.ui.label != "Completed":
                    managed_task.ui.label = "Completed"
                    managed_task.ui.menu.collapsed = True
            elif managed_task.running:
                num_running_tasks += 1
                result = managed_task.most_recent_result_or_none()
                if result:
                    managed_task.status, managed_task.progress = result

    def gl_display(self):
        self._set_progress_icon_indication()

    def _set_progress_icon_indication(self):
        # average progress of all current tasks (including waiting and done)
        num_tasks = len(self.managed_tasks)
        sum_progress = 0
        for managed_task in self.managed_tasks:
            if managed_task.queued:
                sum_progress += 0
            elif managed_task.completed:
                sum_progress += 1.0
            elif managed_task.running:
                sum_progress += managed_task.progress_as_fraction
        progress = sum_progress / num_tasks if num_tasks > 0 else 0.0
        self.menu_icon.indicator_stop = progress

    def _remove_tasks(self, tasks):
        for task in tasks:
            self.managed_tasks.remove(task)
            self.task_container.remove(task.ui.menu)


class TaskUI:
    """
    Wrapper for a submenu showing info and controls for a single task.
    Every task has such a menu.
    """

    def __init__(self, managed_task):
        # it's difficult to override Growing_Menu as this is a Cython class,
        # so we unfortunately have to wrap it
        self.menu = ui.Growing_Menu(managed_task.heading)
        self._label = None
        self._init(managed_task)

    @property
    def label(self):
        """
        A label string (like "queued" or "running") that gets appended to the task title.
        Not to be confused with the status of the ManagedTask.
        """
        return self._label

    @label.setter
    def label(self, label):
        self._remove_old_status_if_exists()
        self._label = label
        self.menu.label += " - " + label

    def _init(self, managed_task):
        self.menu.append(
            ui.Text_Input("status", managed_task, label="Status", setter=lambda x: None)
        )
        progress_bar = ui.Slider(
            "progress",
            managed_task,
            min=managed_task.min_progress,
            max=managed_task.max_progress,
            label="Progress",
        )
        progress_bar.read_only = True
        self.menu.append(progress_bar)
        self.menu.append(ui.Button("Cancel", managed_task.cancel))

    def _remove_old_status_if_exists(self):
        if self._label is not None:
            num_characters_to_remove = len(self._label) + 3
            self.menu.label = self.menu.label[:-num_characters_to_remove]


class ManagedTask:
    """
    Create an instance of this and add it to a task manager via add_task()
    """

    def __init__(self, task, args, heading, min_progress, max_progress):
        """
        :param task: function that will be executed in a new process.
            The function needs to yield tuples (status, progress) where status
            is a string that will be shown to the user and progress is a number
        :param args: tuple with arguments passed to "task"
        :param heading: Task description shown in the UI
        :param min_progress: minimum progress value your task will yield
        :param max_progress: maximum progress value your task will yield
        """
        assert min_progress < max_progress
        self.task = task
        self.args = args
        self.task_proxy = None
        self.heading = heading
        self.status = "Task Queued"
        self.progress = min_progress
        self.min_progress = min_progress
        self.max_progress = max_progress
        self.ui = TaskUI(self)
        self._canceled = False

    @property
    def queued(self):
        return self.task_proxy is None

    @property
    def running(self):
        return (
            self.task_proxy is not None
            and not self.task_proxy.completed
            and not self.task_proxy.canceled
        )

    @property
    def completed(self):
        return self.task_proxy is not None and self.task_proxy.completed

    @property
    def canceled(self):
        # a task can be canceled by canceling the corresponding process or by
        # "canceling" a completed or queued task (in these cases there is no process!)
        process_canceled = self.task_proxy is not None and self.task_proxy.canceled
        return self._canceled or process_canceled

    @property
    def progress_as_fraction(self):
        return (self.progress - self.min_progress) / (
            self.max_progress - self.min_progress
        )

    def start(self):
        assert self.task_proxy is None
        self.task_proxy = bh.IPC_Logging_Task_Proxy(
            self.heading, self.task, args=self.args
        )

    def cancel(self):
        self._canceled = True
        if self.task_proxy is not None:
            self.task_proxy.cancel()
        self.status = "Task Canceled"

    def most_recent_result_or_none(self):
        assert self.running
        result = None
        for result in self.task_proxy.fetch():
            pass
        return result
