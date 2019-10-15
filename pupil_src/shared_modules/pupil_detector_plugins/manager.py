import typing as T

from pyglui import ui

from plugin import Plugin

from . import available_detector_plugins


class PupilDetectorManager(Plugin):
    def __init__(self, g_pool):
        super().__init__(g_pool)

        (
            self._default_pupil_detector_class,
            self._available_pupil_detector_classes,
        ) = available_detector_plugins()

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
        general_settings.insert(0, self._selector)

    def on_notify(self, notification):
        subject = notification["subject"]
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
        plugin_class = self.active_detector.__class__
        plugin_name = self.registry.name_from_class(plugin_class)
        return plugin_name

    def ui_selector_setter(self, value: str):
        self.activate_detector_by_name(plugin_name=value)
        self.active_detector.init_ui()

    @property
    def ui_selector_values(self) -> T.List[str]:
        return self.registry.registered_plugin_names()

    @property
    def ui_selector_labels(self) -> T.List[str]:
        return self.registry.registered_plugin_labels()
