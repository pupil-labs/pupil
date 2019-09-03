"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# Keep source compatibility
from .multiprocessing import EarlyCancellationError
from .multiprocessing import MultiprocessingTaskProxy as Task_Proxy
from .multiprocessing import MultiprocessingLoggingTaskProxy as IPC_Logging_Task_Proxy

from .multiprocessing import MultiprocessingTaskProxy
from .multiprocessing import MultiprocessingLoggingTaskProxy

from .threading import ThreadingTaskProxy
