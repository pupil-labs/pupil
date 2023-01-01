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
            except Exception:
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
        self.optimize_camera_intrinsics = True
        self.open_visualization_window = False
        self.show_camera_trace_in_3d_window = False
        self.render_markers = True
        self.show_marker_id_in_main_window = False
        self.window_size = (1000, 1000)
        self.window_position = (90, 90)

        self.load_from_disk()

    def _data_from_tuple(self, settings_tuple):
        (
            self.detection_frame_index_range,
            self.optimization_frame_index_range,
            self.localization_frame_index_range,
            self.user_defined_origin_marker_id,
            self.optimize_camera_intrinsics,
            self.open_visualization_window,
            self.show_camera_trace_in_3d_window,
            self.render_markers,
            self.show_marker_id_in_main_window,
            self.window_size,
            self.window_position,
        ) = settings_tuple

    @property
    def _data_as_tuple(self):
        return (
            self.detection_frame_index_range,
            self.optimization_frame_index_range,
            self.localization_frame_index_range,
            self.user_defined_origin_marker_id,
            self.optimize_camera_intrinsics,
            self.open_visualization_window,
            self.show_camera_trace_in_3d_window,
            self.render_markers,
            self.show_marker_id_in_main_window,
            self.window_size,
            self.window_position,
        )


class OnlineSettings:
    def __init__(self, settings_tuple):
        (
            self.optimize_markers_3d_model,
            self.optimize_camera_intrinsics,
            self.open_visualization_window,
            self.show_camera_trace_in_3d_window,
            self.render_markers,
            self.show_marker_id_in_main_window,
            self.window_size,
            self.window_position,
        ) = settings_tuple

    @property
    def data_as_dict(self):
        return {
            "optimize_markers_3d_model": self.optimize_markers_3d_model,
            "optimize_camera_intrinsics": self.optimize_camera_intrinsics,
            "open_visualization_window": self.open_visualization_window,
            "show_camera_trace_in_3d_window": self.show_camera_trace_in_3d_window,
            "render_markers": self.render_markers,
            "show_marker_id_in_main_window": self.show_marker_id_in_main_window,
            "window_size": self.window_size,
            "window_position": self.window_position,
        }
