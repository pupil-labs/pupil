"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import typing as T


class EarlyCancellationError(Exception):
    pass


class BaseTaskProxy(abc.ABC):
    """Future like object that runs a given generator in the background and returns is able to return the results incrementally"""

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def __init__(self, name: str, generator, args=(), kwargs={}, **_):
        self._name = name

    @abc.abstractmethod
    def fetch(self) -> T.Iterator[T.Any]:
        """Fetches progress and available results from background"""
        pass

    @abc.abstractmethod
    def cancel(self, timeout=1):
        """
        Cancel current task.

        :param float timeout: Number of seconds to wait for the task to stop its processing.
        """
        pass

    @property
    @abc.abstractmethod
    def completed(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def canceled(self) -> bool:
        pass
