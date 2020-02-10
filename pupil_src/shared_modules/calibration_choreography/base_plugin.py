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
import functools as F
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


class UnsupportedChoreographyModeError(NotImplementedError):
    def __init__(self, mode: ChoreographyMode):
        super().__init__(f"Unknown mode {mode}")


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
            raise ValueError(
                f"Notification missing required keys: {missing_required_keys}"
            )

        valid_keys = cls._REQUIRED_KEYS.union(cls._OPTIONAL_KEYS)
        invalid_keys = keys.difference(valid_keys)
        if invalid_keys:
            raise ValueError(f"Notification contains invalid keys: {invalid_keys}")

        mode, action = note["subject"].split(".")
        return ChoreographyNotification(
            mode=ChoreographyMode(mode), action=ChoreographyAction(action)
        )


class CalibrationChoreographyPlugin(Plugin):
    """Base class for all calibration routines"""

    _THUMBNAIL_COLOR_ON = (0.3, 0.2, 1.0, 0.9)
    __registered_choreography_plugins = {}

    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"
    uniqueness = "by_base_class"

    ### Public

    label = None

    """Controlls wheather the choreography plugin is shown in the user selection list.
    """
    is_user_selectable = True

    """Controlls wheather calibration and accuracy test buttons are visible.
    """
    shows_action_buttons = True

    @classmethod
    @abc.abstractmethod
    def supported_gazer_classes(cls):
        pass

    @classmethod
    def user_selectable_gazer_classes(cls):
        gazer_classes = cls.supported_gazer_classes()
        gazer_classes = sorted(gazer_classes, key=lambda g: g.label)
        return gazer_classes

    @classmethod
    def is_user_selection_for_gazer_enabled(cls) -> bool:
        return len(cls.user_selectable_gazer_classes()) > 1

    @classmethod
    def default_selected_gazer_class(cls):
        return cls.user_selectable_gazer_classes()[0]

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        if cls.label:
            return cls.label
        else:
            raise NotImplementedError(f'{cls} must implement a "label" class property')

    @staticmethod
    def registered_choreographies_by_label() -> T.Mapping[
        str, "CalibrationChoreographyPlugin"
    ]:
        return dict(CalibrationChoreographyPlugin.__registered_choreography_plugins)

    @classmethod
    def user_selectable_choreography_classes(cls):
        choreo_classes = cls.registered_choreographies_by_label().values()
        choreo_classes = filter(lambda c: c.is_user_selectable, choreo_classes)
        choreo_classes = sorted(choreo_classes, key=lambda c: c.label)
        return choreo_classes

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        store = CalibrationChoreographyPlugin.__registered_choreography_plugins
        assert isinstance(
            cls.label, str
        ), f'Calibration choreography plugin subclass {cls.__name__} must overwrite string class property "label"'
        assert (
            cls.label not in store.keys()
        ), f'Calibration choreography plugin already exists for label "{cls.label}"'
        store[cls.label] = cls

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.__is_active = False
        self.__ref_list = []
        self.__pupil_list = []

        self.__current_mode = ChoreographyMode.CALIBRATION
        self.selected_gazer_class = self.default_selected_gazer_class()

        self.__choreography_ui_selector = None
        self.__gazer_ui_selector = None
        self.__ui_button_calibration = None
        self.__ui_button_accuracy_test = None

    def cleanup(self):
        pass

    @property
    def current_mode(self) -> ChoreographyMode:
        return self.__current_mode

    @property
    def is_active(self) -> bool:
        return self.__is_active

    @property
    def selected_choreography_class(self):
        return self.__class__

    @selected_choreography_class.setter
    def selected_choreography_class(self, cls):
        self.__start_plugin(cls)

    @property
    def status_text(self) -> str:
        ui_button = self.__mode_button(self.current_mode)
        return ui_button.status_text or ""

    @status_text.setter
    def status_text(self, value: T.Any):
        value = str(value).strip() if value else ""
        ui_button = self.__mode_button(self.current_mode)
        ui_button.status_text = value

    @property
    def pupil_list(self) -> T.List[dict]:
        return self.__pupil_list

    @property
    def ref_list(self) -> T.List[dict]:
        return self.__ref_list

    def on_choreography_started(self, mode: ChoreographyMode):
        self.notify_all(
            ChoreographyNotification(
                # TODO: Why the subject is always "calibration.started", and not "accuracy_test.started" for accuracy test mode?
                # mode=mode,
                mode=ChoreographyMode.CALIBRATION,
                action=ChoreographyAction.STARTED,
            ).to_dict()
        )

    def on_choreography_stopped(self, mode: ChoreographyMode):
        self.notify_all(
            ChoreographyNotification(
                mode=mode, action=ChoreographyAction.STOPPED
            ).to_dict()
        )

    def on_choreography_successfull(
        self, mode: ChoreographyMode, pupil_list: list, ref_list: list
    ):
        if mode == ChoreographyMode.CALIBRATION:
            calib_data = {"ref_list": ref_list, "pupil_list": pupil_list}
            self.__start_plugin(self.selected_gazer_class, calib_data=calib_data)
        elif mode == ChoreographyMode.ACCURACY_TEST:
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
            print(
                f"===>>> ACCURACY TEST FINISHED: {len(pupil_list)} pupil datums, {len(ref_list)} ref locations"
            )
        else:
            raise UnsupportedChoreographyModeError(mode)

    ### Public - Plugin

    def init_ui(self):

        self.__ui_selector_choreography = ui.Selector(
            "selected_choreography_class",
            self,
            label="Choreography",
            labels=[c.label for c in self.user_selectable_choreography_classes()],
            selection=self.user_selectable_choreography_classes(),
        )

        self.__ui_selector_gazer = ui.Selector(
            "selected_gazer_class",
            self,
            label="Gazer",
            labels=[g.label for g in self.user_selectable_gazer_classes()],
            selection=self.user_selectable_gazer_classes(),
        )

        self.add_menu()
        self.menu.label = self.label
        self.menu.append(self.__ui_selector_choreography)
        self.menu.append(self.__ui_selector_gazer)

        if self.shows_action_buttons:

            def calibration_setter(should_be_on):
                self.__signal_should_toggle_processing(
                    should_be_on=should_be_on, mode=ChoreographyMode.CALIBRATION
                )

            def accuracy_test_setter(should_be_on):
                self.__signal_should_toggle_processing(
                    should_be_on=should_be_on, mode=ChoreographyMode.ACCURACY_TEST
                )

            self.__ui_button_calibration = ui.Thumb(
                "is_active",
                self,
                label="C+",
                hotkey="c",
                setter=calibration_setter,
                on_color=self._THUMBNAIL_COLOR_ON,
            )

            self.__ui_button_accuracy_test = ui.Thumb(
                "is_active",
                self,
                label="T+",
                hotkey="t",
                setter=accuracy_test_setter,
                on_color=self._THUMBNAIL_COLOR_ON,
            )

            self.__toggle_mode_button_visibility(
                is_visible=True, mode=ChoreographyMode.CALIBRATION
            )
            self.__toggle_mode_button_visibility(
                is_visible=True, mode=ChoreographyMode.ACCURACY_TEST
            )

    def update_ui(self):
        self.__ui_selector_gazer.read_only = (
            not self.is_user_selection_for_gazer_enabled()
        )

    def deinit_ui(self):
        """Gets called when the plugin get terminated, either voluntarily or forced.
        """
        if self.is_active:
            self._perform_stop()
        self.remove_menu()
        for mode in ChoreographyMode:
            self.__toggle_mode_button_visibility(is_visible=False, mode=mode)
        self.__choreography_ui_selector = None
        self.__gazer_ui_selector = None
        self.__ui_button_calibration = None
        self.__ui_button_accuracy_test = None

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
                self._perform_start()

        if note.action == ChoreographyAction.SHOULD_STOP:
            if not self.is_active:
                logger.warning(f"{self.current_mode.label} already stopped.")
            else:
                self._perform_stop()

    ### Internal

    def _signal_should_start(self, mode: ChoreographyMode):
        self.notify_all(
            ChoreographyNotification(
                mode=mode, action=ChoreographyAction.SHOULD_START
            ).to_dict()
        )

    def _signal_should_stop(self, mode: ChoreographyMode):
        self.notify_all(
            ChoreographyNotification(
                mode=mode, action=ChoreographyAction.SHOULD_STOP
            ).to_dict()
        )

    def _perform_start(self):
        current_mode = self.__current_mode

        logger.info(f"Starting  {current_mode.label}")

        ### Set the calibration choreography state

        self.__is_active = True
        self.__ref_list = []
        self.__pupil_list = []

        ### Set the calibration choreography UI

        # Hide all buttons for the mode buttons that are not currently used
        for mode in list(ChoreographyMode):
            if self.__current_mode != mode:
                self.__toggle_mode_button_visibility(is_visible=False, mode=mode)

        ### Call relevant callbacks

        self.on_choreography_started(mode=current_mode)

    def _perform_stop(self):
        current_mode = self.__current_mode
        pupil_list = self.__pupil_list
        ref_list = self.__ref_list

        logger.info(f"Stopping  {current_mode.label}")

        ### Set the calibration choreography state

        self.__is_active = False
        self.__ref_list = []
        self.__pupil_list = []

        ### Set the calibration choreography UI

        self.status_text = None

        # Show all buttons for all the modes
        for mode in list(ChoreographyMode):
            self.__toggle_mode_button_visibility(is_visible=True, mode=mode)

        ### Call relevant callbacks

        self.on_choreography_stopped(mode=current_mode)

        if pupil_list and ref_list:
            self.on_choreography_successfull(
                mode=current_mode, pupil_list=pupil_list, ref_list=ref_list
            )

    ### Private

    def __toggle_mode_button_visibility(self, is_visible: bool, mode: ChoreographyMode):
        ui_button = self.__mode_button(mode=mode)

        if ui_button is None:
            return

        if is_visible and (ui_button not in self.g_pool.quickbar):
            # If the button should be visible, but it's not - add to quickbar
            if mode == ChoreographyMode.CALIBRATION:
                self.g_pool.quickbar.insert(0, self.__ui_button_calibration)
            elif mode == ChoreographyMode.ACCURACY_TEST:
                # Always place the accuracy test button first, but after the calibration button
                index = 1 if self.__ui_button_calibration in self.g_pool.quickbar else 0
                self.g_pool.quickbar.insert(index, self.__ui_button_accuracy_test)
            else:
                raise UnsupportedChoreographyModeError(mode)

        if (not is_visible) and (ui_button in self.g_pool.quickbar):
            # If the button should not be visible, but it is - remove from quickbar
            self.g_pool.quickbar.remove(ui_button)

    def __signal_should_toggle_processing(self, should_be_on, mode: ChoreographyMode):
        assert bool(self.is_active) != bool(should_be_on)  # Sanity check
        if should_be_on:
            self._signal_should_start(mode=mode)
        else:
            self._signal_should_stop(mode=mode)

    def __start_plugin(self, plugin_cls, **kwargs):
        self.notify_all(
            {"subject": "start_plugin", "name": plugin_cls.__name__, "args": kwargs}
        )

    def __mode_button(self, mode: ChoreographyMode):
        if mode == ChoreographyMode.CALIBRATION:
            return self.__ui_button_calibration
        if mode == ChoreographyMode.ACCURACY_TEST:
            return self.__ui_button_accuracy_test
        raise UnsupportedChoreographyModeError(mode)
