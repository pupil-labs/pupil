"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import logging
import os

import file_methods as fm

logger = logging.getLogger(__name__)


class SettingsStorage(abc.ABC):
    version = 1

    def __init__(self, save_dir, plugin):
        self._save_dir = save_dir
        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_to_disk()

    def save_to_disk(self):
        self._save_msgpack_to_file(self._msgpack_file_path, self._data_as_tuple)

    def _save_msgpack_to_file(self, file_path, data):
        dict_representation = {"version": self.version, "data": data}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fm.save_object(dict_representation, file_path)

    def load_from_disk(self):
        settings_tuple = self._load_msgpack_from_file(self._msgpack_file_path)
        if settings_tuple:
            try:
                self._data_from_tuple(settings_tuple)
            except:
                pass

    def _load_msgpack_from_file(self, file_path):
        try:
            dict_representation = fm.load_object(file_path)
        except FileNotFoundError:
            return None
        if dict_representation.get("version", None) != self.version:
            logger.warning(
                "Data in {} is in old file format. Will not load these!".format(
                    file_path
                )
            )
            return None
        return dict_representation.get("data", None)

    @property
    def _msgpack_file_name(self):
        return "head_pose_tracker_general_settings.msgpack"

    @property
    def _msgpack_file_path(self):
        return os.path.join(self._save_dir, self._msgpack_file_name)

    @abc.abstractmethod
    def _data_from_tuple(self, settings_tuple):
        pass

    @property
    @abc.abstractmethod
    def _data_as_tuple(self):
        pass


class OfflineSettingsStorage(SettingsStorage):
    def __init__(self, save_dir, plugin, get_recording_index_range):
        save_dir = os.path.join(save_dir, "offline_data")
        super().__init__(save_dir, plugin)

        self._get_recording_index_range = get_recording_index_range

        self.detection_frame_index_range = self._get_recording_index_range()
        self.optimization_frame_index_range = self._get_recording_index_range()
        self.localization_frame_index_range = self._get_recording_index_range()
        self.user_defined_origin_marker_id = None
        self.optimize_camera_intrinsics = False
        self.show_marker_id = False
        self.show_camera_trace = True

        self.load_from_disk()

    def _data_from_tuple(self, settings_tuple):
        (
            self.detection_frame_index_range,
            self.optimization_frame_index_range,
            self.localization_frame_index_range,
            self.user_defined_origin_marker_id,
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.show_camera_trace,
        ) = settings_tuple

    @property
    def _data_as_tuple(self):
        return (
            self.detection_frame_index_range,
            self.optimization_frame_index_range,
            self.localization_frame_index_range,
            self.user_defined_origin_marker_id,
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.show_camera_trace,
        )


class OnlineSettingsStorage(SettingsStorage):
    def __init__(self, save_dir, plugin):
        super().__init__(save_dir, plugin)

        self.optimize_markers_3d_model = True
        self.optimize_camera_intrinsics = False
        self.show_marker_id = False
        self.show_camera_trace = True

        self.load_from_disk()

    def _data_from_tuple(self, settings_tuple):
        (
            self.optimize_markers_3d_model,
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.show_camera_trace,
        ) = settings_tuple

    @property
    def _data_as_tuple(self):
        return (
            self.optimize_markers_3d_model,
            self.optimize_camera_intrinsics,
            self.show_marker_id,
            self.show_camera_trace,
        )
