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
import csv
import json
import logging
import os
import traceback
import typing as T
from collections import namedtuple

import file_methods as fm
import player_methods as pm
import zmq_tools
from hotkey import Hotkey
from plugin import Plugin
from pyglui import ui

logger = logging.getLogger(__name__)


def create_annotation(label, timestamp, duration=0.0, **custom_fields):
    """
    Returns a dictionary in the format needed to send annotations
    to an annotation plugin via the ICP.

    See python/remote_annotations.py in pupil-helpers for an example.

    :param custom_fields:
    """
    return {
        "topic": "annotation",
        "label": label,
        "timestamp": timestamp,
        "duration": duration,
        **custom_fields,
    }


class AnnotationDefinition(T.NamedTuple):
    label: str
    hotkey: str


class AnnotationPlugin(Plugin, abc.ABC):
    """
    Base for player and capture plugins that support adding and removing
    annotations and the corresponding quickbar buttons
    """

    _AnnotationButtons = namedtuple("_AnnotationButtons", "quickbar menu")

    icon_chr = chr(0xE866)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, annotation_definitions=None):
        super().__init__(g_pool)
        self.menu = None
        self._annotation_list_menu = None

        if annotation_definitions is None:
            annotation_definitions = [
                ["My annotation", Hotkey.ANNOTATION_EVENT_DEFAULT_HOTKEY()]
            ]
        self._initial_annotation_definitions = annotation_definitions
        self._definition_to_buttons = {}

        self._new_annotation_label = "new annotation label"
        self._new_annotation_hotkey = Hotkey.ANNOTATION_EVENT_DEFAULT_HOTKEY()

    @property
    def annotation_definitions(self):
        return tuple(self._definition_to_buttons.keys())

    def get_init_dict(self):
        return {"annotation_definitions": self.annotation_definitions}

    def init_ui(self):
        self.add_menu()
        self.customize_menu()
        self.menu.append(
            ui.Text_Input("_new_annotation_label", self, label="New Label")
        )
        self.menu.append(
            ui.Text_Input("_new_annotation_hotkey", self, label="New Hotkey")
        )
        self.menu.append(
            ui.Button("Add Annotation Type", self._on_add_annotation_clicked)
        )
        self._annotation_list_menu = ui.Growing_Menu(
            "Annotation List - Click to Remove"
        )
        self.menu.append(self._annotation_list_menu)
        self._create_initial_annotation_list()

    def _create_initial_annotation_list(self):
        for label, hotkey in self._initial_annotation_definitions:
            self._add_annotation_definition(label, hotkey)

    @abc.abstractmethod
    def customize_menu(self):
        pass

    def deinit_ui(self):
        self._clear_buttons_quickbar()
        self.remove_menu()

    def _clear_buttons_quickbar(self):
        # only call this from deinit_ui()
        for buttons in self._definition_to_buttons.values():
            self._remove_button_quickbar(buttons.quickbar)

    @abc.abstractmethod
    def fire_annotation(self, annotation_definition):
        pass

    def _on_add_annotation_clicked(self):
        # new_annotation_label and hotkey are set by the ui input fields
        self._add_annotation_definition(
            self._new_annotation_label, self._new_annotation_hotkey
        )

    def _add_annotation_definition(self, annotation_label, hotkey):
        annotation_definition = AnnotationDefinition(
            label=annotation_label, hotkey=hotkey
        )
        if annotation_definition in self._definition_to_buttons:
            logger.warning(
                "Cannot add duplicate annotation definition {} <{}>".format(
                    annotation_label, hotkey
                )
            )
            return
        button_quickbar = self._create_button_quickbar(annotation_definition)
        button_menu = self._create_button_menu(annotation_definition)
        annotation_ui = self._AnnotationButtons(button_quickbar, button_menu)
        self._definition_to_buttons[annotation_definition] = annotation_ui

        self._append_button_quickbar(button_quickbar)
        self._append_button_menu(button_menu)

    def _create_button_quickbar(self, annotation_definition):
        def make_fire(_):
            self.fire_annotation(annotation_definition.label)

        return ui.Thumb(
            annotation_definition.label,
            setter=make_fire,
            getter=lambda: False,
            label=annotation_definition.hotkey,
            hotkey=annotation_definition.hotkey,
        )

    def _create_button_menu(self, annotation_definition):
        def make_remove():
            self._remove_annotation_info(annotation_definition)

        label = annotation_definition.label
        hotkey = annotation_definition.hotkey
        return ui.Button(label=f"{label} <{hotkey}>", function=make_remove)

    def _append_button_quickbar(self, button_quickbar):
        current_buttons = self.g_pool.quickbar.elements
        index_of_last_button = -1
        for definition, buttons in self._definition_to_buttons.items():
            try:
                idx = current_buttons.index(buttons.quickbar)
            except ValueError:
                pass
            else:
                index_of_last_button = max(idx, index_of_last_button)
        if index_of_last_button != -1:
            self.g_pool.quickbar.insert(index_of_last_button + 1, button_quickbar)
        else:
            self.g_pool.quickbar.append(button_quickbar)

    def _append_button_menu(self, button_menu):
        self._annotation_list_menu.append(button_menu)

    def _remove_annotation_info(self, annotation_definition):
        buttons = self._definition_to_buttons[annotation_definition]
        self._remove_button_quickbar(buttons.quickbar)
        self._remove_button_menu(buttons.menu)
        del self._definition_to_buttons[annotation_definition]

    def _remove_button_quickbar(self, button_quickbar):
        self.g_pool.quickbar.remove(button_quickbar)

    def _remove_button_menu(self, button_menu):
        self._annotation_list_menu.remove(button_menu)


class Annotation_Capture(AnnotationPlugin):
    """
    Pupil Capture plugin to record annotations.
    """

    def __init__(self, g_pool, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        self.ipc_pub = zmq_tools.Msg_Streamer(
            self.g_pool.zmq_ctx, self.g_pool.ipc_pub_url
        )
        self.annotation_sub = zmq_tools.Msg_Receiver(
            self.g_pool.zmq_ctx, self.g_pool.ipc_sub_url, topics=("annotation",)
        )

    def customize_menu(self):
        self.menu.label = "View and Record Annotations"

    def fire_annotation(self, annotation_label):
        ts = self.g_pool.get_timestamp()
        new_annotation = create_annotation(annotation_label, ts)
        new_annotation["added_in_capture"] = True
        self.ipc_pub.send(new_annotation)

    def recent_events(self, events):
        recent_annotation_data = []
        while self.annotation_sub.new_data:
            topic, annotation_datum = self.annotation_sub.recv()
            ts = self.g_pool.get_timestamp()
            annotation_desc = self._annotation_description(
                label=annotation_datum["label"], age=ts - annotation_datum["timestamp"]
            )
            logger.info(annotation_desc)
            recent_annotation_data.append(annotation_datum)
        events["annotation"] = recent_annotation_data

    @staticmethod
    def _annotation_description(label, age) -> str:
        return f"{label} annotation ({age:.3f} seconds ago)"


class Annotation_Player(AnnotationPlugin, Plugin):
    """
    Pupil Player plugin to view, edit, and add annotations.
    """

    _FILE_DEFINITIONS_VERSION = 1
    _FILE_DEFINITIONS_NAME = "annotation_definitions.json"

    class VersionMismatchError(ValueError):
        pass

    def __init__(self, g_pool, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        str_use_session_settings = " Using session settings."
        try:
            definitions = self.deserialize_definitions_from_recording()
            logger.debug(f"Using annotation definitions from recording: {definitions}")
            self._initial_annotation_definitions = definitions
        except FileNotFoundError:
            logger.debug("No annotation definitions found." + str_use_session_settings)
        except Annotation_Player.VersionMismatchError as err:
            logger.warning(str(err) + str_use_session_settings)
        except (json.JSONDecodeError, AttributeError, KeyError):
            logger.warning(
                "Could not read annotation definitions." + str_use_session_settings
            )
            logger.debug(traceback.format_exc())

        self.annotations = self.load_annotations("annotation_player")
        no_or_empty_player_data_file = len(self.annotations) == 0
        if no_or_empty_player_data_file:
            self.annotations = self.load_annotations("annotation")
        self.last_frame_ts = None
        self.last_frame_index = -1

    def load_annotations(self, file_name):
        annotation_pldata = fm.load_pldata_file(self.g_pool.rec_dir, file_name)
        annotations = pm.Mutable_Bisector(
            annotation_pldata.data, annotation_pldata.timestamps
        )
        logger.info(f"Loaded {len(annotations)} annotations from {file_name}.pldata")
        return annotations

    def cleanup(self):
        with fm.PLData_Writer(self.g_pool.rec_dir, "annotation_player") as writer:
            for ts, annotation in zip(self.annotations.timestamps, self.annotations):
                writer.append_serialized(ts, "annotation", annotation.serialized)
        self.serialize_definitions_to_recording()

    def customize_menu(self):
        self.menu.label = "View and Edit Annotations"
        self.menu.append(
            ui.Info_Text(
                "Annotations recorded with capture are displayed when this "
                "plugin is loaded. New annotations can be added with the "
                "interface below."
            )
        )
        self.menu.append(
            ui.Info_Text(
                "If you want to revert annotations to the recorded state, "
                "stop player, delete the annotation_player.pldata file in the "
                "recording and reopen player."
            )
        )

    def fire_annotation(self, annotation_label):
        if self.last_frame_ts is None:
            return
        if self.last_frame_index < 0:
            return
        ts = self.last_frame_ts
        annotation_desc = self._annotation_description(
            label=annotation_label, world_index=self.last_frame_index
        )
        logger.info(annotation_desc)
        new_annotation = create_annotation(annotation_label, ts)
        new_annotation["added_in_player"] = True
        self.annotations.insert(
            new_annotation["timestamp"], fm.Serialized_Dict(python_dict=new_annotation)
        )

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return
        self.last_frame_ts = frame.timestamp
        if frame.index != self.last_frame_index:
            self.last_frame_index = frame.index
            frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
            events = self.annotations.by_ts_window(frame_window)
            for event in events:
                annotation_desc = self._annotation_description(
                    label=event["label"], world_index=frame.index
                )
                logger.info(annotation_desc)

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.export_annotations(
                notification["ts_window"], notification["export_dir"]
            )

    def export_annotations(self, export_window, export_dir):
        annotation_section = self.annotations.init_dict_for_window(export_window)
        annotation_idc = pm.find_closest(
            self.g_pool.timestamps, annotation_section["data_ts"]
        )
        csv_keys = self.parse_csv_keys(annotation_section["data"])

        with open(
            os.path.join(export_dir, "annotations.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(csv_keys)
            for annotation, idx in zip(annotation_section["data"], annotation_idc):
                csv_row = [idx]
                csv_row.extend(annotation.get(k, "") for k in csv_keys[1:])
                csv_writer.writerow(csv_row)
            logger.info("Created 'annotations.csv' file.")

    @staticmethod
    def parse_csv_keys(annotations):
        csv_keys = ("index", "timestamp", "label", "duration")
        system_keys = set(csv_keys)
        user_keys = set()
        for annotation in annotations:
            # selects keys that are not included in system_keys and
            # adds them to user_keys if they were not included before
            user_keys |= set(annotation.keys()) - system_keys

        user_keys.discard("topic")  # topic is always "annotation"

        # return tuple with system keys first and alphabetically sorted
        # user keys afterwards
        return csv_keys + tuple(sorted(user_keys))

    @staticmethod
    def _annotation_description(label, world_index) -> str:
        return f"{label} annotation @ frame index {world_index}"

    @property
    def file_definitions_path(self):
        return os.path.join(
            self.g_pool.rec_dir, "offline_data", self._FILE_DEFINITIONS_NAME
        )

    def deserialize_definitions_from_recording(self) -> T.Tuple[T.Tuple[str, str], ...]:
        """Read annotation definitions from a json file.

        The file is expected to be at `self.file_definitions_path`

        Raises:
            FileNotFoundError: Default, if the recording is being opened for the first
              time
            json.JSONDecodeError: If the file cannot be parsed
            VersionMisMatchError: If the file has an unexpected format version
            KeyError: If the `definitions` field is not present
            AttributeError: If the `definitions` field is not a mapping

        Returns:
            T.Tuple[T.Tuple[str, str], ...]: Tuple of annotation definitions, each being
              a label-hotkey tuple.
        """
        with open(self.file_definitions_path) as json_file:
            return self._deserialize_definitions_from_file(
                readable_json_file=json_file,
                expected_version=self._FILE_DEFINITIONS_VERSION,
            )

    @staticmethod
    def _deserialize_definitions_from_file(
        readable_json_file, expected_version: int
    ) -> T.Tuple[T.Tuple[str, str], ...]:
        content = json.load(readable_json_file)
        content_version = content.get("version", "unspecified")
        if content_version != expected_version:
            raise Annotation_Player.VersionMismatchError(
                f"Version mismatch: Encountered {content_version}, "
                f"expected {expected_version}"
            )
        content_definitions: T.Dict[str, str] = content["definitions"]
        return tuple(content_definitions.items())

    def serialize_definitions_to_recording(self):
        """Write annotation definitions to a json file.

        The file is expected to be at `self.file_definitions_path`
        """
        with open(self.file_definitions_path, "w") as writable_file:
            self._serialize_definitions_to_file(
                writable_file,
                definitions=self.annotation_definitions,
                version=self._FILE_DEFINITIONS_VERSION,
            )

    @staticmethod
    def _serialize_definitions_to_file(
        writable_file, definitions: T.Iterable[T.Tuple[str, str]], version: int
    ):
        definitions = {label: hotkey for label, hotkey in definitions}
        content = {"version": version, "definitions": definitions}
        json.dump(content, writable_file)
