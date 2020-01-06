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

    def init_ui(self):
        general_settings = self.g_pool.menubar[0]
        self._selector = ui.Selector(
            "pupil_detector",
            getter=self.ui_selector_getter,
            setter=self.ui_selector_setter,
            selection=self.ui_selector_values,
            labels=self.ui_selector_labels,
            label="Detection method",
        )
        general_settings.append(self._selector)

    def on_notify(self, notification):
        subject = notification["subject"]
        if subject in self._notification_handler:
            handler = self._notification_handler[subject]
            handler(notification)

    def set_detection_mode(self, notification):
        mode = notification["mode"]
        self._selector.read_only = mode != "disabled"

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

    # UI Selector

    def ui_selector_getter(self) -> str:
        return self.g_pool.pupil_detector.identifier

    def ui_selector_setter(self, value: str):
        self.activate_detector_by_name(plugin_name=value)
        self.active_detector.init_ui()

    @property
    def ui_selector_values(self) -> T.List[str]:
        return [klass.identifier for klass in self._available_pupil_detector_classes]

    @property
    def ui_selector_labels(self) -> T.List[str]:
        return [klass.label for klass in self._available_pupil_detector_classes]
