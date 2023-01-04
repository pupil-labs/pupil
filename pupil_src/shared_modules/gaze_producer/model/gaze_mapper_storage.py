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

import file_methods as fm
import make_unique
from gaze_producer import model
from observable import Observable
from storage import SingleFileStorage

logger = logging.getLogger(__name__)


class GazeMapperStorage(SingleFileStorage, Observable):
    def __init__(self, calibration_storage, rec_dir, get_recording_index_range):
        super().__init__(rec_dir)
        self._calibration_storage = calibration_storage
        self._get_recording_index_range = get_recording_index_range
        self._gaze_mappers = []
        self._load_from_disk()
        if not self._gaze_mappers:
            self._add_default_gaze_mapper()

    def _add_default_gaze_mapper(self):
        self.add(self.create_default_gaze_mapper())

    def create_default_gaze_mapper(self):
        default_calibration = self._calibration_storage.get_first_or_none()
        if default_calibration:
            calibration_unique_id = default_calibration.unique_id
        else:
            calibration_unique_id = ""
        return model.GazeMapper(
            unique_id=model.GazeMapper.create_new_unique_id(),
            name=make_unique.by_number_at_end("Default Gaze Mapper", self.item_names),
            calibration_unique_id=calibration_unique_id,
            mapping_index_range=self._get_recording_index_range(),
            validation_index_range=self._get_recording_index_range(),
            validation_outlier_threshold_deg=5.0,
        )

    def duplicate_gaze_mapper(self, gaze_mapper):
        return model.GazeMapper(
            unique_id=gaze_mapper.create_new_unique_id(),
            name=make_unique.by_number_at_end(
                gaze_mapper.name + " Copy", self.item_names
            ),
            calibration_unique_id=gaze_mapper.calibration_unique_id,
            mapping_index_range=gaze_mapper.mapping_index_range,
            validation_index_range=gaze_mapper.validation_index_range,
            validation_outlier_threshold_deg=gaze_mapper.validation_outlier_threshold_deg,
            manual_correction_x=gaze_mapper.manual_correction_x,
            manual_correction_y=gaze_mapper.manual_correction_y,
            activate_gaze=gaze_mapper.activate_gaze,
            # We cannot deep copy gaze, so we don't.
            # All others left at their default.
        )

    def add(self, gaze_mapper):
        self._gaze_mappers.append(gaze_mapper)
        self._gaze_mappers.sort(key=lambda g: g.name)

    def delete(self, gaze_mapper):
        self._gaze_mappers.remove(gaze_mapper)
        self._delete_mapping_file(gaze_mapper)

    def _delete_mapping_file(self, gaze_mapper):
        mapping_file_path = self._gaze_mapping_file_path(gaze_mapper)
        try:
            os.remove(mapping_file_path + ".pldata")
            os.remove(mapping_file_path + "_timestamps.npy")
        except FileNotFoundError:
            pass

    def rename(self, gaze_mapper, new_name):
        old_mapping_file_path = self._gaze_mapping_file_path(gaze_mapper)
        gaze_mapper.name = new_name
        new_mapping_file_path = self._gaze_mapping_file_path(gaze_mapper)
        self._rename_mapping_file(old_mapping_file_path, new_mapping_file_path)

    def _rename_mapping_file(self, old_mapping_file_path, new_mapping_file_path):
        try:
            os.rename(
                old_mapping_file_path + ".pldata", new_mapping_file_path + ".pldata"
            )
            os.rename(
                old_mapping_file_path + "_timestamps.npy",
                new_mapping_file_path + "_timestamps.npy",
            )
        except FileNotFoundError:
            pass

    def save_to_disk(self):
        # this will save everything except gaze and gaze_ts
        super().save_to_disk()

        self._save_gaze_and_ts_to_disk()

    def _save_gaze_and_ts_to_disk(self):
        directory = self._gaze_mappings_directory
        os.makedirs(directory, exist_ok=True)
        for gaze_mapper in self._gaze_mappers:
            file_name = self._gaze_mapping_file_name(gaze_mapper)
            with fm.PLData_Writer(directory, file_name) as writer:
                for gaze_ts, gaze in zip(gaze_mapper.gaze_ts, gaze_mapper.gaze):
                    writer.append_serialized(
                        gaze_ts, topic="gaze", datum_serialized=gaze.serialized
                    )

    def _load_from_disk(self):
        # this will load everything except gaze and gaze_ts
        super()._load_from_disk()

        self._load_gaze_and_ts_from_disk()

    def _load_gaze_and_ts_from_disk(self):
        directory = self._gaze_mappings_directory
        for gaze_mapper in self._gaze_mappers:
            file_name = self._gaze_mapping_file_name(gaze_mapper)
            pldata = fm.load_pldata_file(directory, file_name)
            gaze_mapper.gaze = pldata.data
            gaze_mapper.gaze_ts = pldata.timestamps

    @property
    def _storage_file_name(self):
        return "gaze_mappers.msgpack"

    @property
    def _item_class(self):
        return model.GazeMapper

    @property
    def items(self):
        return self._gaze_mappers

    @property
    def item_names(self):
        return [gaze_mapper.name for gaze_mapper in self._gaze_mappers]

    @property
    def _gaze_mappings_directory(self):
        return os.path.join(self._storage_folder_path, "gaze-mappings")

    def _gaze_mapping_file_name(self, gaze_mapper):
        file_name = gaze_mapper.name + "-" + gaze_mapper.unique_id
        return self.get_valid_filename(file_name)

    def _gaze_mapping_file_path(self, gaze_mapper):
        return os.path.join(
            self._gaze_mappings_directory, self._gaze_mapping_file_name(gaze_mapper)
        )
