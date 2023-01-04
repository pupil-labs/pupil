"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)

Contains common observers for tasks.

"""

import logging
import traceback

logger = logging.getLogger(__name__)


def raise_exception(exception):
    """
    If you don't expect exceptions in your task, add this as an observer to
    "on_exception" for tasks.
    """
    if exception:
        # TODO Ensure the exception's traceback is forwarded correctly
        raise exception
    else:
        stack_trace_str = "".join(traceback.format_stack())
        logger.error(
            "A background task raised an unknown exception! You might find the "
            "causing task in this stacktrace:\n" + stack_trace_str
        )
