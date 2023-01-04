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
import enum
import functools as F
import logging
import typing as T

import audio
from gaze_mapping import GazerHMD3D, default_gazer_class, registered_gazer_classes
from gaze_mapping.gazer_base import GazerBase
from hotkey import Hotkey
from plugin import Plugin

logger = logging.getLogger(__name__)


class ChoreographyMode(enum.Enum):
    CALIBRATION = "calibration"
    VALIDATION = "validation"

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
    ADD_REF_DATA = "add_ref_data"
    DATA = "data"


class ChoreographyNotification:
    __slots__ = ("mode", "action", "payload")

    _REQUIRED_KEYS = {"subject"}

    def __init__(self, mode: ChoreographyMode, action: ChoreographyAction, **payload):
        self.mode = mode
        self.action = action
        self.payload = payload

    @property
    def subject(self) -> str:
        return f"{self.mode.value}.{self.action.value}"

    def __getitem__(self, key: str) -> T.Any:
        return self.payload.__getitem__(key)

    def get(self, key: str, default_value) -> T.Any:
        return self.payload.get(key, default_value)

    def to_dict(self) -> dict:
        return {"subject": self.subject, **self.payload}

    @staticmethod
    def from_dict(
        note: dict, allow_extra_keys: bool = False
    ) -> "ChoreographyNotification":
        cls = ChoreographyNotification
        note = note.copy()
        keys = set(note.keys())

        missing_required_keys = cls._REQUIRED_KEYS.difference(keys)
        if missing_required_keys:
            raise ValueError(
                f"Notification missing required keys: {missing_required_keys}"
            )

        mode, action = note["subject"].split(".")
        del note["subject"]
        return ChoreographyNotification(
            mode=ChoreographyMode(mode), action=ChoreographyAction(action), **note
        )


class CalibrationChoreographyPlugin(Plugin):
    """Base class for all calibration routines"""

    _THUMBNAIL_COLOR_ON = (0.3, 0.2, 1.0, 0.9)
    __registered_choreography_plugins = {}

    order = 0.3
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

    """Controlls wheather the choreography plugin is persistent across sessions.
    """
    is_session_persistent = True

    @classmethod
    def selection_label(cls) -> str:
        return cls.label

    @classmethod
    def selection_order(cls) -> float:
        return float("inf")

    @classmethod
    def supported_gazer_classes(cls):
        gazers = registered_gazer_classes()
        # By default, HMD gazers are not supported by regular choreographies
        gazers = [g for g in gazers if not issubclass(g, GazerHMD3D)]
        return gazers

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
        if default_gazer_class in cls.user_selectable_gazer_classes():
            return default_gazer_class
        else:
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
        # First sort alphabetically by selection_label, then sort by selection_order
        choreo_classes = sorted(choreo_classes, key=lambda c: c.selection_label())
        choreo_classes = sorted(choreo_classes, key=lambda c: c.selection_order())
        return choreo_classes

    @classmethod
    def should_register(cls) -> bool:
        return True

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if not cls.should_register():
            # If the class label is explicitly saying that it shouldn't be registered,
            # Skip the class registration; this is usefull for abstract superclasses.
            return

        store = CalibrationChoreographyPlugin.__registered_choreography_plugins

        assert isinstance(
            cls.label, str
        ), f'Calibration choreography plugin subclass {cls.__name__} must overwrite string class property "label"'

        assert (
            cls.label not in store.keys()
        ), f'Calibration choreography plugin already exists for label "{cls.label}"'

        store[cls.label] = cls

    def __init__(self, g_pool, selected_gazer_class_name=None):
        if selected_gazer_class_name is not None:
            supported_gazers_by_name = {
                g.__name__: g for g in self.supported_gazer_classes()
            }
            selected_gazer_class = supported_gazers_by_name.get(
                selected_gazer_class_name, None
            )
            if selected_gazer_class is None:
                logger.debug(
                    f'Selected gazer class "{selected_gazer_class_name}" is not supported by "{self.__class__.__name__}" choreography'
                )
        else:
            selected_gazer_class = None

        super().__init__(g_pool)

        self.__is_active = False
        self.__ref_list = []
        self.__pupil_list = []

        self.__current_mode = ChoreographyMode.CALIBRATION
        self.selected_gazer_class = selected_gazer_class

        self.__choreography_ui_selector = None
        self.__gazer_ui_selector = None
        self.__ui_button_calibration = None
        self.__ui_button_validation = None

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
        self._start_plugin(
            cls, selected_gazer_class_name=self.selected_gazer_class.__name__
        )

    @property
    def selected_gazer_class(self):
        return self.__selected_gazer_class

    @selected_gazer_class.setter
    def selected_gazer_class(self, cls):
        if (cls is None) or (cls not in self.supported_gazer_classes()):
            default_cls = self.default_selected_gazer_class()
            if cls is not None:
                logger.debug(
                    f'Selected gazer "{cls.__name__}" not supported by {self.__class__.__name__} choreography; using default gazer "{default_cls.__name__}"'
                )
            cls = default_cls
        self.__selected_gazer_class = cls
        self._update_gazer_description_ui_text()

    @property
    def status_text(self) -> str:
        ui_button = self.__mode_button(self.current_mode)
        return ui_button.status_text or ""

    @status_text.setter
    def status_text(self, value: T.Any):
        value = str(value).strip() if value else ""
        ui_button = self.__mode_button(self.current_mode)
        if ui_button:
            ui_button.status_text = value

    @property
    def pupil_list(self) -> T.List[dict]:
        return self.__pupil_list

    @property
    def ref_list(self) -> T.List[dict]:
        return self.__ref_list

    @pupil_list.setter
    def pupil_list(self, value):
        self.__pupil_list = value

    @ref_list.setter
    def ref_list(self, value):
        self.__ref_list = value

    def on_choreography_started(self, mode: ChoreographyMode):
        self.notify_all(
            ChoreographyNotification(
                mode=mode,
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
            self._start_plugin(self.selected_gazer_class, calib_data=calib_data)
        elif mode == ChoreographyMode.VALIDATION:
            assert self.g_pool.active_gaze_mapping_plugin is not None
            gazer_class = self.g_pool.active_gaze_mapping_plugin.__class__
            gazer_params = self.g_pool.active_gaze_mapping_plugin.get_params()

            self._start_plugin("Accuracy_Visualizer")
            self.notify_all(
                ChoreographyNotification(
                    mode=ChoreographyMode.VALIDATION,
                    action=ChoreographyAction.DATA,
                    gazer_class_name=gazer_class.__name__,
                    gazer_params=gazer_params,
                    pupil_list=pupil_list,
                    ref_list=ref_list,
                    timestamp=self.g_pool.get_timestamp(),
                    record=True,
                ).to_dict()
            )
        else:
            raise UnsupportedChoreographyModeError(mode)

    ### Public - Plugin

    @classmethod
    def base_class(cls):
        # This ensures that all choreography plugins return the same base class,
        # even choreographies that subclass concrete choreography implementations.
        return CalibrationChoreographyPlugin

    @classmethod
    def _choreography_description_text(cls) -> str:
        return ""

    def _init_custom_menu_ui_elements(self) -> list:
        return []

    def _update_gazer_description_ui_text(self):
        try:
            ui_text = self.__ui_gazer_description_text
        except AttributeError:
            return
        ui_text.text = self.selected_gazer_class._gazer_description_text()

    def init_ui(self):
        from pyglui import ui

        desc_text = ui.Info_Text(self._choreography_description_text())

        self.__ui_selector_choreography = ui.Selector(
            "selected_choreography_class",
            self,
            label="Choreography",
            selection_getter=self.__choreography_selection_getter,
        )

        self.__ui_selector_gazer = ui.Selector(
            "selected_gazer_class",
            self,
            label="Gaze Mapping",
            labels=[g.label for g in self.user_selectable_gazer_classes()],
            selection=self.user_selectable_gazer_classes(),
        )

        self.__ui_gazer_description_text = ui.Info_Text("")
        self._update_gazer_description_ui_text()

        best_practices_text = ui.Info_Text(
            "Read more about best practices at docs.pupil-labs.com"
        )

        custom_ui_elements = self._init_custom_menu_ui_elements()

        super().init_ui()
        self.add_menu()
        self.menu.label = self.label
        self.menu_icon.order = self.order
        self.menu_icon.tooltip = "Calibration"

        # Construct menu UI
        self.menu.append(self.__ui_selector_choreography)
        self.menu.append(desc_text)
        if len(custom_ui_elements) > 0:
            self.menu.append(ui.Separator())
            for ui_elem in custom_ui_elements:
                self.menu.append(ui_elem)
            self.menu.append(ui.Separator())
        else:
            self.menu.append(ui.Separator())
        self.menu.append(self.__ui_selector_gazer)
        self.menu.append(self.__ui_gazer_description_text)
        self.menu.append(best_practices_text)

        if self.shows_action_buttons:

            def calibration_setter(should_be_on):
                self.__signal_should_toggle_processing(
                    should_be_on=should_be_on, mode=ChoreographyMode.CALIBRATION
                )

            def validation_setter(should_be_on):
                self.__signal_should_toggle_processing(
                    should_be_on=should_be_on, mode=ChoreographyMode.VALIDATION
                )

            self.__ui_button_calibration = ui.Thumb(
                "is_active",
                self,
                label="C",
                hotkey=Hotkey.GAZE_CALIBRATION_CAPTURE_HOTKEY(),
                setter=calibration_setter,
                on_color=self._THUMBNAIL_COLOR_ON,
            )

            self.__ui_button_validation = ui.Thumb(
                "is_active",
                self,
                label="T",
                hotkey=Hotkey.GAZE_VALIDATION_CAPTURE_HOTKEY(),
                setter=validation_setter,
                on_color=self._THUMBNAIL_COLOR_ON,
            )

            self.__toggle_mode_button_visibility(
                is_visible=True, mode=ChoreographyMode.CALIBRATION
            )
            self.__toggle_mode_button_visibility(
                is_visible=True, mode=ChoreographyMode.VALIDATION
            )

    def update_ui(self):
        self.__ui_selector_gazer.read_only = (
            not self.is_user_selection_for_gazer_enabled()
        )
        if self.shows_action_buttons:
            self.__ui_button_validation.read_only = (
                self.g_pool.active_gaze_mapping_plugin is None
            )

    def deinit_ui(self):
        """Gets called when the plugin get terminated, either voluntarily or forced."""
        if self.is_active:
            self._perform_stop()
        self.remove_menu()
        for mode in ChoreographyMode:
            self.__toggle_mode_button_visibility(is_visible=False, mode=mode)
        self.__choreography_ui_selector = None
        self.__gazer_ui_selector = None
        self.__ui_button_calibration = None
        self.__ui_button_validation = None

    def recent_events(self, events):
        if self.g_pool.app == "capture":
            # UI is only initialized in Capture. In other applications, i.e. Service,
            # calling this function will crash with an AttributeError.
            self.update_ui()

    def on_notify(self, note_dict):
        """Handles choreography notifications

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``: Stops the calibration procedure
           ``calibration.add_ref_data``: Adds reference data

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
            return  # Unknown/unexpected notification, not handling it

        if note.action == ChoreographyAction.SHOULD_START:
            if self.is_active:
                logger.warning(f"{self.current_mode.label} already running.")
            else:
                self.__current_mode = note.mode
                self._perform_start()

        if note.action == ChoreographyAction.ADD_REF_DATA:
            if self.is_active:
                pass  # No-op; each choreography should handle it independently
            else:
                logger.error("Ref data can only be added when calibration is running.")

        if note.action == ChoreographyAction.SHOULD_STOP:
            if not self.is_active:
                logger.debug(f"{self.current_mode.label} already stopped.")
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
        if self.__is_active:
            logger.debug(
                "[PROGRAMMING ERROR] Called _perform_start on an already active "
                "calibration choreography."
            )
            return

        current_mode = self.__current_mode

        logger.info(f"Starting  {current_mode.label}")
        audio.tink()

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
        if not self.__is_active:
            logger.debug(
                "[PROGRAMMING ERROR] Called _perform_stop on an already inactive "
                "calibration choreography."
            )
            return

        if self.g_pool.app == "capture":
            # Reset the main window size to trigger a redraw with correct size and scale
            # Only run in Capture to fix https://github.com/pupil-labs/pupil/issues/2119
            self.g_pool.trigger_main_window_redraw()

        current_mode = self.__current_mode
        pupil_list = self.__pupil_list
        ref_list = self.__ref_list

        logger.info(f"Stopping  {current_mode.label}")
        audio.tink()

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

        self.on_choreography_successfull(
            mode=current_mode, pupil_list=pupil_list, ref_list=ref_list
        )

    def _start_plugin(self, plugin_cls_or_name, **kwargs):
        if isinstance(plugin_cls_or_name, object.__class__):
            plugin_name = plugin_cls_or_name.__name__
        elif isinstance(plugin_cls_or_name, str):
            plugin_name = plugin_cls_or_name
        else:
            raise ValueError(
                f"Expected instance of type or str, but got {plugin_cls_or_name.__class__.__name__}"
            )
        self.notify_all(
            {"subject": "start_plugin", "name": plugin_name, "args": kwargs}
        )

    ### Private

    @classmethod
    def __choreography_selection_getter(cls):
        selection = cls.user_selectable_choreography_classes()
        labels = [c.selection_label() for c in selection]
        return selection, labels

    def __toggle_mode_button_visibility(self, is_visible: bool, mode: ChoreographyMode):
        ui_button = self.__mode_button(mode=mode)

        if ui_button is None:
            return

        if is_visible and (ui_button not in self.g_pool.quickbar):
            # If the button should be visible, but it's not - add to quickbar
            if mode == ChoreographyMode.CALIBRATION:
                self.g_pool.quickbar.insert(0, self.__ui_button_calibration)
            elif mode == ChoreographyMode.VALIDATION:
                # Always place the accuracy test button first, but after the calibration button
                index = 1 if self.__ui_button_calibration in self.g_pool.quickbar else 0
                self.g_pool.quickbar.insert(index, self.__ui_button_validation)
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

    def __mode_button(self, mode: ChoreographyMode):
        if mode == ChoreographyMode.CALIBRATION:
            return self.__ui_button_calibration
        if mode == ChoreographyMode.VALIDATION:
            return self.__ui_button_validation
        raise UnsupportedChoreographyModeError(mode)
