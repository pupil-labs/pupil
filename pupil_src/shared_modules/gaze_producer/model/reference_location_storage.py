"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import os
from dataclasses import MISSING, fields
from types import SimpleNamespace
from typing import Dict, Tuple

import file_methods as fm
import numpy as np
import numpy.typing as npt
from camera_models import Camera_Model
from gaze_mapping.notifications import CalibrationSetupNotification
from gaze_producer import model
from observable import Observable
from player_methods import find_closest
from storage import SingleFileStorage
from video_capture.file_backend import File_Source

logger = logging.getLogger(__name__)


class ReferenceLocationStorage(SingleFileStorage, Observable):
    def __init__(self, rec_dir: str):
        super().__init__(rec_dir)
        self._reference_locations: Dict[int, model.ReferenceLocation] = {}
        self._load_from_disk()
        if not self._reference_locations:
            self._load_from_recorded_notifications()

    def _load_from_recorded_notifications(self):
        notifications = fm.load_pldata_file(self._rec_dir, "notify")
        intrinsics, timestamps = self._load_world_intrinsics_and_timestamps()
        logger.debug(f"Using {intrinsics} to project eventual 3d reference data")
        for topic, data in zip(notifications.topics, notifications.data):
            if topic.startswith("notify."):
                # Remove "notify." prefix
                data = data._deep_copy_dict()
                data["subject"] = data["topic"][len("notify.") :]
                del data["topic"]
            else:
                continue
            if (
                CalibrationSetupNotification.calibration_format_version()
                != model.Calibration.version
            ):
                logger.debug(
                    f"Must update CalibrationResultNotification to match Calibration version"
                )
                continue
            try:
                note = CalibrationSetupNotification.from_dict(data)
            except ValueError as err:
                logger.debug(str(err))
                continue

            references = note.calib_data["ref_list"]
            ref_ts = [ref["timestamp"] for ref in references]
            world_idc = find_closest(timestamps, ref_ts).tolist()
            for reference, idx in zip(references, world_idc):
                reference["frame_index"] = idx
                self._add_recorded_reference(intrinsics, reference)

    def _add_recorded_reference(self, intrinsics: Camera_Model, reference) -> None:
        if "mm_pos" in reference:
            # Unity uses a left-handed coordinate system while Pupil Core software
            # assumes a right-handed coordinate system. This is important to get the
            # 3d bundle adjustment correct as well as the projection into the 2d plane.
            mm_pos = np.array(reference["mm_pos"])
            mm_pos[1] *= -1.0

            reference["mm_pos"] = tuple(mm_pos.tolist())
            reference["screen_pos"] = tuple(
                intrinsics.projectPoints(mm_pos).reshape(-1).tolist()
            )
        ref_fields = fields(model.ReferenceLocation)
        ref_loc = {  # required fields
            f.name: reference[f.name] for f in ref_fields if f.default is MISSING
        }
        ref_loc.update(  # optional fields
            {
                f.name: reference.get(f.name, f.default)
                for f in ref_fields
                if f.default is not MISSING
            }
        )
        self.add(model.ReferenceLocation(**ref_loc))

    def _load_world_intrinsics_and_timestamps(
        self,
    ) -> Tuple[Camera_Model, npt.NDArray[np.float64]]:
        g_pool = SimpleNamespace()
        source = File_Source(g_pool, source_path=os.path.join(self._rec_dir, "world"))
        return source.intrinsics, source.timestamps

    def add(self, reference_location: model.ReferenceLocation):
        self._reference_locations[reference_location.frame_index] = reference_location

    def get_or_none(self, frame_index):
        return self._reference_locations.get(frame_index, None)

    def get_next(self, frame_index):
        found_index = min(
            idx for idx in self._reference_locations.keys() if idx > frame_index
        )
        return self._reference_locations[found_index]

    def get_previous(self, frame_index):
        found_index = max(
            idx for idx in self._reference_locations.keys() if idx < frame_index
        )
        return self._reference_locations[found_index]

    def get_in_range(self, frame_index_range):
        def in_range(ref):
            return frame_index_range[0] <= ref.frame_index <= frame_index_range[1]

        return [ref for ref in self._reference_locations.values() if in_range(ref)]

    def delete(self, reference_location):
        del self._reference_locations[reference_location.frame_index]

    def delete_all(self):
        self._reference_locations.clear()

    @property
    def _storage_file_name(self):
        return "reference_locations.msgpack"

    @property
    def _item_class(self):
        return model.ReferenceLocation

    @property
    def items(self):
        return self._reference_locations.values()
