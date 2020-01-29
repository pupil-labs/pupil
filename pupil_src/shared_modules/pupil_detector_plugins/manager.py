"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as T

from pyglui import ui

from plugin import Plugin

from . import available_detector_plugins


class PupilDetectorManager(Plugin):
    order = 0.2

    def __init__(self, g_pool):
        super().__init__(g_pool)

        _, self._available_pupil_detector_classes = available_detector_plugins()

        self._notification_handler = {
            "set_detection_mapping_mode": self.set_detection_mode
        }

    def on_notify(self, notification):
        subject = notification["subject"]
        if subject in self._notification_handler:
            handler = self._notification_handler[subject]
            handler(notification)

    def set_detection_mode(self, notification):
        mode = notification["mode"]

        for detector_cls in self._available_pupil_detector_classes:
            if detector_cls.identifier == mode:
                props = notification.get("properties", None)
                self._notify_new_detector(detector_cls, properties=props)
                return

    def _notify_new_detector(self, detector_cls, properties=None):
        self.notify_all(
            {
                "subject": "start_eye_plugin",
                "target": self.g_pool.process,
                "name": detector_cls.__name__,
                "args": properties or {},
            }
        )
