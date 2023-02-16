"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import inspect

from tasklib.background.patches import IPCLoggingPatch, KeyboardInterruptHandlerPatch
from tasklib.background.task import BackgroundGeneratorFunction, BackgroundRoutine


def create(
    name,
    routine_or_generator_function,
    pass_shared_memory=False,
    args=None,
    kwargs=None,
    patches=None,
):
    """
    Creates the right background task for your type of task.

    Normally, you would not use this directly, but use a PluginTaskManager and the
    create_background_task() method there.
    If you don't use a manager, you need to regularly call update() on the returned task
    to trigger callbacks etc.

    See the docstring for PluginTaskManager.create_background_task()
    (in tasklib.manager.py) for information about the different parameters!
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if patches is None:
        patches = [
            IPCLoggingPatch(),
            KeyboardInterruptHandlerPatch(),
        ]

    if inspect.isgeneratorfunction(routine_or_generator_function):
        return BackgroundGeneratorFunction(
            name,
            routine_or_generator_function,
            pass_shared_memory,
            args,
            kwargs,
            patches,
        )
    elif inspect.isroutine(routine_or_generator_function):
        return BackgroundRoutine(
            name,
            routine_or_generator_function,
            pass_shared_memory,
            args,
            kwargs,
            patches,
        )
    else:
        raise TypeError(
            "Cannot create background task from {}. It must be a "
            "routine (function, method, lambda) or generator "
            "function!".format(routine_or_generator_function)
        )
