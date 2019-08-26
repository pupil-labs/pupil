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
from .multiprocessing import EarlyCancellationError, Task_Proxy, IPC_Logging_Task_Proxy

from .threading import Task_Proxy as Threading_Task_Proxy

from .multiprocessing import Task_Proxy as Multiprocessing_Task_Proxy
from .multiprocessing import IPC_Logging_Task_Proxy as Multiprocessing_Logging_Task_Proxy
