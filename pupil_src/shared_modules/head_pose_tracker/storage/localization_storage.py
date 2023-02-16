"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections
import os

import file_methods as fm
import numpy as np
import player_methods as pm
from observable import Observable


class Localization:
    def __init__(self):
        self.set_to_default_values()

    def set_to_default_values(self):
        self.recent_camera_trace = collections.deque(maxlen=300)

    def add_recent_camera_trace(self, camera_trace):
        self.recent_camera_trace.append(camera_trace)

    @property
    def none_pose_data(self):
        return {
            "camera_extrinsics": None,
            "camera_poses": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "camera_trace": [np.nan, np.nan, np.nan],
            "camera_pose_matrix": None,
        }


class OfflineCameraLocalization(Localization):
    def __init__(self, get_current_frame_index, get_current_frame_window):
        super().__init__()

        self._get_current_frame_index = get_current_frame_index
        self._get_current_frame_window = get_current_frame_window

    def set_to_default_values(self):
        super().set_to_default_values()
        self.pose_bisector = pm.Mutable_Bisector()

    @property
    def calculated(self):
        return bool(self.pose_bisector)

    @property
    def current_pose(self):
        frame_window = self._get_current_frame_window()
        try:
            pose_data = self.pose_bisector.by_ts_window(frame_window)[0]
        except IndexError:
            return self.none_pose_data
        else:
            return pose_data


class OfflineLocalizationStorage(Observable, OfflineCameraLocalization):
    def __init__(
        self, rec_dir, plugin, get_current_frame_index, get_current_frame_window
    ):
        super().__init__(get_current_frame_index, get_current_frame_window)

        self._rec_dir = rec_dir

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
            for pose_ts, pose in zip(
                self.pose_bisector.timestamps, self.pose_bisector.data
            ):
                writer.append_serialized(
                    pose_ts, topic="pose", datum_serialized=pose.serialized
                )

    def load_pldata_from_disk(self):
        self._load_from_file()

    def _load_from_file(self):
        directory = self._offline_data_folder_path
        file_name = self._pldata_file_name
        pldata = fm.load_pldata_file(directory, file_name)
        self.pose_bisector = pm.Mutable_Bisector(pldata.data, pldata.timestamps)

    @property
    def _offline_data_folder_path(self):
        return os.path.join(self._rec_dir, "offline_data")

    @property
    def _pldata_file_name(self):
        return "camera_pose"


class OnlineLocalizationStorage(Localization):
    def __init__(self):
        super().__init__()

        self.current_pose = self.none_pose_data
