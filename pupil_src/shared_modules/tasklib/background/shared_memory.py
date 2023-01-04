"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import multiprocessing as mp
from ctypes import c_bool, c_double


class SharedMemory:
    """
    Contains information shared between the foreground and a background process.
    All properties are process safe.
    """

    def __init__(self):
        self._should_terminate_flag = mp.Value(c_bool, False)
        self._progress = mp.Value(c_double, 0.0)

    @property
    def should_terminate_flag(self):
        """
        Boolean flag indicating if the background was asked to shut itself down.
        """
        return self._should_terminate_flag.value

    @should_terminate_flag.setter
    def should_terminate_flag(self, new_flag):
        self._should_terminate_flag.value = new_flag

    @property
    def progress(self):
        """
        A float value containing the progress of the background task.

        Usually, this should be a value from 0 to 1. However, you are free to set it
        to whatever numbers you like. Just be aware that integers will also be
        converted to float. If you set it to e.g. 21003, the actual value will be
        something like 21002.9999859.
        """
        return self._progress.value

    @progress.setter
    def progress(self, new_progress):
        self._progress.value = new_progress
