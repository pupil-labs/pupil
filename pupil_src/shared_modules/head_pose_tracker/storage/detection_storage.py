"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os

import file_methods as fm
import player_methods as pm
from observable import Observable


class OfflineMarkerLocation:
    def __init__(self, get_current_frame_index, get_current_frame_window):
        self._get_current_frame_index = get_current_frame_index
        self._get_current_frame_window = get_current_frame_window

        self.markers_bisector = pm.Mutable_Bisector()
        self.frame_index_to_num_markers = {}

    @property
    def calculated(self):
        return bool(self.markers_bisector)

    @property
    def current_markers(self):
        frame_index = self._get_current_frame_index()
        try:
            num_markers = self.frame_index_to_num_markers[frame_index]
        except KeyError:
            num_markers = 0

        if num_markers:
            frame_window = self._get_current_frame_window()
            return self.markers_bisector.by_ts_window(frame_window)
        else:
            return []


class OfflineDetectionStorage(Observable, OfflineMarkerLocation):
    def __init__(
        self,
        rec_dir,
        all_timestamps,
        plugin,
        get_current_frame_index,
        get_current_frame_window,
    ):
        super().__init__(get_current_frame_index, get_current_frame_window)

        self._rec_dir = rec_dir
        self._all_timestamps = all_timestamps.tolist()

        self.load_pldata_from_disk()

        plugin.add_observer("cleanup", self._on_cleanup)

    def _on_cleanup(self):
        self.save_pldata_to_disk()

    def save_pldata_to_disk(self):
        self._save_to_file()

    def _save_to_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        os.makedirs(directory, exist_ok=True)
        with fm.PLData_Writer(directory, file_name) as writer:
            for marker_ts, marker in zip(
                self.markers_bisector.timestamps, self.markers_bisector.data
            ):
                writer.append_serialized(
                    timestamp=marker_ts, topic="", datum_serialized=marker.serialized
                )
        self._save_frame_index_to_num_markers()

    def _save_frame_index_to_num_markers(self):
        directory = self._offline_data_folder_path
        file_name = self._frame_index_to_num_markers_file_name
        path = os.path.join(directory, file_name)
        fm.save_object(self.frame_index_to_num_markers, path)

    def load_pldata_from_disk(self):
        self._load_from_file()

    def _load_from_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        pldata = fm.load_pldata_file(directory, file_name)
        self.markers_bisector = pm.Mutable_Bisector(pldata.data, pldata.timestamps)

        if pldata.topics and pldata.topics[0] == "":
            self._load_frame_index_to_num_markers()
        else:
            # for backward compatibility
            for topic in set(pldata.topics):
                frame_index, num_markers = topic.split(".")
                self.frame_index_to_num_markers[int(frame_index)] = int(num_markers)

    def _load_frame_index_to_num_markers(self):
        directory = self._offline_data_folder_path
        file_name = self._frame_index_to_num_markers_file_name
        path = os.path.join(directory, file_name)
        try:
            self.frame_index_to_num_markers = fm.load_object(path)
        except FileNotFoundError:
            pass

    @property
    def _pldata_file_name(self):
        return "marker_detection"

    @property
    def _frame_index_to_num_markers_file_name(self):
        return "frame_index_to_num_markers.msgpack"

    @property
    def _offline_data_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")


class OnlineDetectionStorage:
    def __init__(self):
        self.current_markers = []
