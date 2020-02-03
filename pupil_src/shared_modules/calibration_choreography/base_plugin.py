"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import enum
import logging
import typing as T

from pyglui import ui
from plugin import Plugin


logger = logging.getLogger(__name__)


# TODO: Move to gazer module
class Gazer(Plugin, abc.ABC):
    label = None

    @staticmethod
    def registered_gazers() -> T.Mapping[str, "Gazer"]:
        return dict(Gazer.__registered_choreography_plugins)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        try:
            store = Gazer.__registered_choreography_plugins
        except AttributeError:
            Gazer.__registered_choreography_plugins = {}
            store = Gazer.__registered_choreography_plugins
        assert isinstance(cls.label, str), f"Gazer subclass {cls.__name__} must overwrite string class property \"label\""
        store[cls.label] = cls


# TODO: Move to gazer module
class Gazer2D(Gazer):
    label = "2D"


# TODO: Move to gazer module
class Gazer3D(Gazer):
    label = "3D"


class GazeDimensionality(enum.Enum):
    GAZE_2D = "2d"
    GAZE_3D = "3d"

    @property
    def label(self) -> str:
        return self.value().upper()


class ChoreographyMode(enum.Enum):
    CALIBRATION = "calibration"
    ACCURACY_TEST = "accuracy_test"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class ChoreograthyAction(enum.Enum):
    SHOULD_START = "should_start"
    SHOULD_STOP = "should_stop"
    STARTED = "started"
    STOPPED = "stopped"
    FAILED = "failed"
    SUCCEEDED = "successful"

class ChoreograthyNotification:
    __slots__ = ("mode", "action")

    _REQUIRED_KEYS = {"subject"}
    _OPTIONAL_KEYS = {"topic"}

    def __init__(self, mode: ChoreographyMode, action: ChoreograthyAction):
        self.mode = mode
        self.action = action

    @property
    def subject(self) -> str:
        return f"{self.mode.value}.{self.action.value}"

    def to_dict(self) -> dict:
        return {"subject": self.subject}

    @staticmethod
    def from_dict(note: dict) -> "ChoreographyNotification":
        cls = ChoreograthyNotification
        keys = set(note.keys())

        missing_required_keys = cls._REQUIRED_KEYS.difference(keys)
        if missing_required_keys:
            raise ValueError(f"Notification missing required keys: {missing_required_keys}")

        valid_keys = cls._REQUIRED_KEYS.union(cls._OPTIONAL_KEYS)
        invalid_keys = keys.difference(valid_keys)
        if invalid_keys:
            raise ValueError(f"Notification contains invalid keys: {invalid_keys}")

        mode, action = note["subject"].split(".")
        return ChoreograthyNotification(
            mode=ChoreographyMode(mode),
            action=ChoreograthyAction(action),
        )


class CalibrationChoreographyPlugin(Plugin):
    """base class for all calibration routines"""

    _THUMBNAIL_COLOR_ON = (0.3, 0.2, 1.0, 0.9)
    __registered_choreography_plugins = {}

    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"
    uniqueness = "by_base_class"

    label = None

    """Controlls wheather the choreography plugin is shown in the user selection list.
    """
    is_user_selectable = True

    """Controlls wheather calibration and accuracy test buttons are visible.
    """
    shows_action_buttons = True

    @abc.abstractmethod
    def gazer_for_dimensionality(self, dimensionality: GazeDimensionality):
        pass

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return cls.label

    @staticmethod
    def registered_choreographies() -> T.Mapping[str, "CalibrationChoreographyPlugin"]:
        return dict(CalibrationChoreographyPlugin.__registered_choreography_plugins)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        store = CalibrationChoreographyPlugin.__registered_choreography_plugins
        assert isinstance(cls.label, str), f"Calibration choreography plugin subclass {cls.__name__} must overwrite string class property \"label\""
        assert cls.label not in store.keys(), f"Calibration choreography plugin already exists for label \"{cls.label}\""
        store[cls.label] = cls

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.__is_active = False
        self.__current_mode = ChoreographyMode.CALIBRATION
        self.__choreography_ui_selector = None
        self.__dimensionality_ui_selector = None
        self.__selected_dimensionality = self.__ui_selector_dimensionality_selection()[0]

    def cleanup(self):
        pass

    @property
    def current_mode(self) -> ChoreographyMode:
        return self.__current_mode

    @property
    def current_mode_ui_button(self):
        if self.__current_mode == ChoreographyMode.CALIBRATION:
            return self.__ui_button_calibration
        if self.__current_mode == ChoreographyMode.ACCURACY_TEST:
            return self.__ui_button_accuracy_test
        raise NotImplementedError(f"Unsupported choreography mode: {self.__current_mode}")

    @property
    def is_active(self) -> bool:
        return self.__is_active

    def start(self):
        self.__is_active = True
        if self.__current_mode == ChoreographyMode.CALIBRATION:
            self.__ui_button_accuracy_test_disable()
        if self.__current_mode == ChoreographyMode.ACCURACY_TEST:
            self.__ui_button_calibation_disable()
        self.notify_all(
            ChoreograthyNotification(
                # TODO: Why the subject is always "calibration.started", and not "accuracy_test.started" for accuracy test mode?
                # mode=self.__current_mode,
                mode=ChoreographyMode.CALIBRATION,
                action=ChoreograthyAction.STARTED,
            ).to_dict()
        )

    def stop(self):
        self.__is_active = False
        self.__ui_button_calibation_enable()
        self.__ui_button_accuracy_test_enable()
        self.notify_all(
            ChoreograthyNotification(
                mode=self.__current_mode,
                action=ChoreograthyAction.STOPPED,
            ).to_dict()
        )

    # TODO: Replace with a callback
    def finish_calibration(self, pupil_list, ref_list):
        # finish_calibration(self.g_pool, pupil_list, ref_list)
        print(f"===>>> CALIBRATION FINISHED: {len(pupil_list)} pupil datums, {len(ref_list)} ref locations")

    # TODO: Replace with a callback
    def finish_accuracy_test(self, pupil_list, ref_list):
        # ts = self.g_pool.get_timestamp()
        # self.notify_all({"subject": "start_plugin", "name": "Accuracy_Visualizer"})
        # self.notify_all(
        #     {
        #         "subject": "accuracy_test.data",
        #         "timestamp": ts,
        #         "pupil_list": pupil_list,
        #         "ref_list": ref_list,
        #         "record": True,
        #     }
        # )
        print(f"===>>> ACCURACY TEST FINISHED: {len(pupil_list)} pupil datums, {len(ref_list)} ref locations")

    # Private

    def __available_dimensions_and_gazers(self):
        dimensions = sorted(GazeDimensionality, key=lambda dim: dim.value)
        gazers = map(self.gazer_for_dimensionality, dimensions)
        dimensions_and_gazers = zip(dimensions, gazers)
        dimensions_and_gazers = filter(lambda pair: pair[1] is not None, dimensions_and_gazers)
        dimensions_and_gazers = list(dimensions_and_gazers)
        return dimensions_and_gazers

    def __available_choreography_labels_and_plugins(self):
        labels_and_plugins = CalibrationChoreographyPlugin.registered_choreographies().items()
        labels_and_plugins = filter(lambda pair: pair[1].is_user_selectable, labels_and_plugins)
        labels_and_plugins = sorted(labels_and_plugins, key=lambda pair: pair[0])
        return labels_and_plugins

    def __toggle_mode_action(self, mode: ChoreographyMode):
        action = ChoreograthyAction.SHOULD_START if not self.is_active else ChoreograthyAction.SHOULD_STOP
        self.notify_all(
            ChoreograthyNotification(
                mode=mode, action=action
            ).to_dict()
        )

    def __start_plugin(self, plugin_cls, **kwargs):
        self.notify_all({"subject": "start_plugin", "name": plugin_cls.__name__, "args": kwargs})

    # Public - Plugin

    def init_ui(self):

        self.__ui_selector_choreography = ui.Selector(
            "choreography_selector",
            label="Choreography",
            labels=self.__ui_selector_choreography_labels(),
            selection=self.__ui_selector_choreography_selection(),
            getter=self.__ui_selector_choreography_getter,
            setter=self.__ui_selector_choreography_setter,
        )

        self.__ui_selector_dimensionality = ui.Selector(
            "dimensionality_selector",
            label="Dimensionality",
            labels=self.__ui_selector_dimensionality_labels(),
            selection=self.__ui_selector_dimensionality_selection(),
            getter=self.__ui_selector_dimensionality_getter,
            setter=self.__ui_selector_dimensionality_setter,
        )

        self.__ui_button_calibration = ui.Thumb(
            "is_active",
            self,
            label="C+",
            hotkey="c",
            setter=self.__ui_button_calibration_toggle,
            on_color=self._THUMBNAIL_COLOR_ON,
        )

        self.__ui_button_accuracy_test = ui.Thumb(
            "is_active",
            self,
            label="T+",
            hotkey="t",
            setter=self.__ui_button_accuracy_test_toggle,
            on_color=self._THUMBNAIL_COLOR_ON
        )

        self.add_menu()
        self.menu.label = self.label
        self.menu.append(self.__ui_selector_choreography)
        self.menu.append(self.__ui_selector_dimensionality)

        self.__ui_button_calibation_enable()
        self.__ui_button_accuracy_test_enable()

    def update_ui(self):
        self.__ui_selector_dimensionality.read_only = not self.__ui_selector_dimensionality_enabled()

    def deinit_ui(self):
        self.__ui_button_calibation_disable()
        self.__ui_button_accuracy_test_disable()
        self.remove_menu()
        self.__ui_button_calibration = None
        self.__ui_button_accuracy_test = None
        self.__ui_selector_choreography = None
        self.__ui_selector_dimensionality = None

    def recent_events(self, events):
        self.update_ui()

    def on_notify(self, note_dict):
        """Handles choreography notifications

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``: Stops the calibration procedure

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped
            ``calibration.failed``: Calibration failed
            ``calibration.successful``: Calibration succeeded

        Args:
            note_dict (dict): Notification dictionary
        """
        try:
            note = ChoreograthyNotification.from_dict(note_dict)
        except ValueError:
            return  # Disregard notifications other than choreography notifications

        if note.action == ChoreograthyAction.SHOULD_START:
            if self.is_active:
                logger.warning(f"{self.current_mode.label} already running.")
            else:
                self.__current_mode = note.mode
                self.start()

        if note.action == ChoreograthyAction.SHOULD_STOP:
            if not self.is_active:
                logger.warning(f"{self.current_mode.label} already stopped.")
            else:
                self.stop()

    # Private - UI

    def __ui_selector_choreography_labels(self):
        return [label for label, _ in self.__available_choreography_labels_and_plugins()]

    def __ui_selector_choreography_selection(self):
        return [plugin_cls for _, plugin_cls in self.__available_choreography_labels_and_plugins()]

    def __ui_selector_choreography_getter(self):
        return self.__class__

    def __ui_selector_choreography_setter(self, value):
        self.__start_plugin(value)

    def __ui_selector_dimensionality_enabled(self):
        return len(self.__ui_selector_dimensionality_labels()) > 1

    def __ui_selector_dimensionality_labels(self):
        return [dimension.label for dimension, _ in self.__available_dimensions_and_gazers()]

    def __ui_selector_dimensionality_selection(self):
        return [gazer for _, gazer in self.__available_dimensions_and_gazers()]

    def __ui_selector_dimensionality_getter(self):
        return self.__selected_dimensionality

    def __ui_selector_dimensionality_setter(self, value):
        self.__selected_dimensionality = value

    def __ui_button_calibration_toggle(self, _=None):
        self.__toggle_mode_action(mode=ChoreographyMode.CALIBRATION)

    def __ui_button_calibation_enable(self):
        if not self.shows_action_buttons:
            return
        if self.__ui_button_calibration not in self.g_pool.quickbar:
            self.g_pool.quickbar.insert(0, self.__ui_button_calibration)

    def __ui_button_calibation_disable(self):
        if self.__ui_button_calibration in self.g_pool.quickbar:
            self.g_pool.quickbar.remove(self.__ui_button_calibration)

    def __ui_button_accuracy_test_toggle(self, _=None):
        self.__toggle_mode_action(mode=ChoreographyMode.ACCURACY_TEST)

    def __ui_button_accuracy_test_enable(self):
        if not self.shows_action_buttons:
            return
        if self.__ui_button_accuracy_test not in self.g_pool.quickbar:
            # Always place the accuracy test button first, but after the calibration button
            index = 1 if self.__ui_button_calibration in self.g_pool.quickbar else 0
            self.g_pool.quickbar.insert(index, self.__ui_button_accuracy_test)

    def __ui_button_accuracy_test_disable(self):
        if self.__ui_button_accuracy_test in self.g_pool.quickbar:
            self.g_pool.quickbar.remove(self.__ui_button_accuracy_test)
