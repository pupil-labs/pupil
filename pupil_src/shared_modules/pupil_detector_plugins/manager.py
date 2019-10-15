import typing as T
from plugin import Plugin


class PupilDetectorManager(Plugin):
    def __init__(self, g_pool):
        from pupil_detectors.plugins import (
            PupilDetectorPluginRegistry,
            DetectorDummyPlugin,
            Detector2DPlugin,
            Detector3DPlugin,
        )

        self.g_pool = g_pool
        self.registry = PupilDetectorPluginRegistry.shared_registry()

        # Set manager defaults
        self._default_pupil_detector_class = Detector2DPlugin
        self._default_pupil_detector_class_for_mapping_mode_2d = Detector2DPlugin
        self._default_pupil_detector_class_for_mapping_mode_3d = Detector3DPlugin
        self._default_pupil_detector_class_for_mapping_mode_none = DetectorDummyPlugin

    @property
    def active_detector_name(self) -> str:
        plugin_class = self.active_detector.__class__
        plugin_name = self.registry.name_from_class(plugin_class)
        return plugin_name

    @property
    def active_detector(self):
        try:
            return self.g_pool.pupil_detector
        except AttributeError:
            return None

    @active_detector.setter
    def active_detector(self, detector):
        if self.active_detector:
            self.active_detector.deinit_ui()
            self.active_detector.cleanup()
        self.g_pool.pupil_detector = detector
        # NOTE: Need to init_ui manually

    def activate_detector_by_name(
        self, plugin_name: str, g_pool=..., plugin_properties=None
    ):
        plugin_class = self.registry.class_by_name(plugin_name)
        self.activate_detector_by_class(plugin_class=plugin_class, g_pool=g_pool)

    def activate_detector_by_class(
        self, plugin_class, g_pool=..., plugin_properties=None
    ):
        g_pool = g_pool if g_pool is not ... else self.g_pool
        self.active_detector = plugin_class(g_pool, plugin_properties)

    # Session persistance

    _PUPIL_DETECTOR_SELECTION_NAME_KEY = "last_pupil_detector"
    _PUPIL_DETECTOR_PROPERTIES_KEY = "pupil_detector_settings"

    def load_from_session_settings(self, session_settings: T.Mapping[str, T.Any]):
        plugin_name = session_settings.get(
            self._PUPIL_DETECTOR_SELECTION_NAME_KEY, None
        )
        plugin_prop = session_settings.get(self._PUPIL_DETECTOR_PROPERTIES_KEY, None)
        if plugin_name:
            self.activate_detector_by_name(
                plugin_name=plugin_name, plugin_properties=plugin_prop
            )
        else:
            plugin_class = self._default_pupil_detector_class
            self.activate_detector_by_class(
                plugin_class=plugin_class, plugin_properties=plugin_prop
            )

    def save_to_session_settings(self, session_settings: T.Mapping[str, T.Any]):
        plugin_name = self.active_detector_name
        plugin_prop = self.active_detector.namespaced_detector_properties()
        session_settings[self._PUPIL_DETECTOR_SELECTION_NAME_KEY] = plugin_name
        session_settings[self._PUPIL_DETECTOR_PROPERTIES_KEY] = plugin_prop

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

    # Mapping Mode Selector

    def set_detection_mapping_mode_2d(self):
        self._set_detection_mapping_mode_with_class(
            plugin_class=self._default_pupil_detector_class_for_mapping_mode_2d
        )

    def set_detection_mapping_mode_3d(self):
        self._set_detection_mapping_mode_with_class(
            plugin_class=self._default_pupil_detector_class_for_mapping_mode_3d
        )

    def set_detection_mapping_mode_none(self):
        self._set_detection_mapping_mode_with_class(
            plugin_class=self._default_pupil_detector_class_for_mapping_mode_none
        )

    def _set_detection_mapping_mode_with_class(self, plugin_class):
        if not isinstance(self.active_detector, plugin_class):
            self.activate_detector_by_class(plugin_class=plugin_class)
            self.active_detector.init_ui()
