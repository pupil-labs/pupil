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


class _BaseTask:
    @property
    @abc.abstractmethod
    def progress(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def process(self):
        pass

    @abc.abstractmethod
    def cancel(self):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass

    def on_started(self):
        pass

    def on_updated(self, data):
        pass

    def on_canceled(self):
        pass

    def on_failed(self, error):
        pass

    def on_completed(self, data):
        pass
