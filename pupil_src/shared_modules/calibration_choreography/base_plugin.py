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

from gaze_mapping.gazer_base import GazerBase


logger = logging.getLogger(__name__)


class ChoreographyMode(enum.Enum):
    CALIBRATION = "calibration"
    ACCURACY_TEST = "accuracy_test"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class ChoreographyAction(enum.Enum):
    SHOULD_START = "should_start"
    SHOULD_STOP = "should_stop"
    STARTED = "started"
    STOPPED = "stopped"
    FAILED = "failed"
    SUCCEEDED = "successful"

class ChoreographyNotification:
    __slots__ = ("mode", "action")

    _REQUIRED_KEYS = {"subject"}
    _OPTIONAL_KEYS = {"topic"}

    def __init__(self, mode: ChoreographyMode, action: ChoreographyAction):
        self.mode = mode
        self.action = action

    @property
    def subject(self) -> str:
        return f"{self.mode.value}.{self.action.value}"

    def to_dict(self) -> dict:
        return {"subject": self.subject}

    @staticmethod
    def from_dict(note: dict) -> "ChoreographyNotification":
        cls = ChoreographyNotification
        keys = set(note.keys())

        missing_required_keys = cls._REQUIRED_KEYS.difference(keys)
        if missing_required_keys:
            raise ValueError(f"Notification missing required keys: {missing_required_keys}")

        valid_keys = cls._REQUIRED_KEYS.union(cls._OPTIONAL_KEYS)
        invalid_keys = keys.difference(valid_keys)
        if invalid_keys:
            raise ValueError(f"Notification contains invalid keys: {invalid_keys}")

        mode, action = note["subject"].split(".")
        return ChoreographyNotification(
            mode=ChoreographyMode(mode),
            action=ChoreographyAction(action),
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
    def supported_gazers(self):
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
        self.__gazer_ui_selector = None
        self.__selected_gazer = self.__ui_selector_gazer_selection()[0]

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
    def current_gazer(self) -> GazerBase:
        return self.__selected_gazer

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
            ChoreographyNotification(
                # TODO: Why the subject is always "calibration.started", and not "accuracy_test.started" for accuracy test mode?
                # mode=self.__current_mode,
                mode=ChoreographyMode.CALIBRATION,
                action=ChoreographyAction.STARTED,
            ).to_dict()
        )

    def stop(self):
        self.__is_active = False
        self.__ui_button_calibation_enable()
        self.__ui_button_accuracy_test_enable()
        self.notify_all(
            ChoreographyNotification(
                mode=self.__current_mode,
                action=ChoreographyAction.STOPPED,
            ).to_dict()
        )

    # TODO: Replace with a callback
    def finish_calibration(self, pupil_list, ref_list):
        calib_data = {"ref_list": ref_list, "pupil_list": pupil_list}
        self.__start_plugin(self.current_gazer, calib_data=calib_data)

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

    def __available_choreography_labels_and_plugins(self):
        labels_and_plugins = CalibrationChoreographyPlugin.registered_choreographies().items()
        labels_and_plugins = filter(lambda pair: pair[1].is_user_selectable, labels_and_plugins)
        labels_and_plugins = sorted(labels_and_plugins, key=lambda pair: pair[0])
        return labels_and_plugins

    def __available_gazers_labels_and_plugins(self):
        pairs = [(g.label, g) for g in self.supported_gazers()]
        pairs = sorted(pairs, key=lambda pair: pair[0])
        return pairs

    def __toggle_mode_action(self, mode: ChoreographyMode):
        action = ChoreographyAction.SHOULD_START if not self.is_active else ChoreographyAction.SHOULD_STOP
        self.notify_all(
            ChoreographyNotification(
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

        self.__ui_selector_gazer = ui.Selector(
            "gazer_selector",
            label="Gazer",
            labels=self.__ui_selector_gazer_labels(),
            selection=self.__ui_selector_gazer_selection(),
            getter=self.__ui_selector_gazer_getter,
            setter=self.__ui_selector_gazer_setter,
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
        self.menu.append(self.__ui_selector_gazer)

        self.__ui_button_calibation_enable()
        self.__ui_button_accuracy_test_enable()

    def update_ui(self):
        self.__ui_selector_gazer.read_only = not self.__ui_selector_gazer_enabled()

    def deinit_ui(self):
        self.__ui_button_calibation_disable()
        self.__ui_button_accuracy_test_disable()
        self.remove_menu()
        self.__ui_button_calibration = None
        self.__ui_button_accuracy_test = None
        self.__ui_selector_choreography = None
        self.__ui_selector_gazer = None

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
            note = ChoreographyNotification.from_dict(note_dict)
        except ValueError:
            return  # Disregard notifications other than choreography notifications

        if note.action == ChoreographyAction.SHOULD_START:
            if self.is_active:
                logger.warning(f"{self.current_mode.label} already running.")
            else:
                self.__current_mode = note.mode
                self.start()

        if note.action == ChoreographyAction.SHOULD_STOP:
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

    def __ui_selector_gazer_enabled(self):
        return len(self.__ui_selector_gazer_labels()) > 1

    def __ui_selector_gazer_labels(self):
        return [label for label, _ in self.__available_gazers_labels_and_plugins()]

    def __ui_selector_gazer_selection(self):
        return [gazer for _, gazer in self.__available_gazers_labels_and_plugins()]

    def __ui_selector_gazer_getter(self):
        return self.__selected_gazer

    def __ui_selector_gazer_setter(self, value):
        self.__selected_gazer = value

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
