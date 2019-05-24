"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import copy
import logging
import os

import file_methods as fm
import make_unique

from storage import Storage
from gaze_producer import model
from observable import Observable

logger = logging.getLogger(__name__)


class CalibrationStorage(Storage, Observable):
    _calibration_suffix = "plcal"

    def __init__(self, rec_dir, plugin, get_recording_index_range, recording_uuid):
        super().__init__(plugin)
        self._rec_dir = rec_dir
        self._get_recording_index_range = get_recording_index_range
        self._recording_uuid = recording_uuid
        self._calibrations = []
        self._load_from_disk()
        if not self._calibrations:
            self._add_default_calibration()

    def _add_default_calibration(self):
        self.add(self.create_default_calibration())

    def create_default_calibration(self):
        return model.Calibration(
            unique_id=model.Calibration.create_new_unique_id(),
            name=make_unique.by_number_at_end("Default Calibration", self.item_names),
            recording_uuid=self._recording_uuid,
            mapping_method="3d",
            frame_index_range=self._get_recording_index_range(),
            minimum_confidence=0.8,
        )

    def duplicate_calibration(self, calibration):
        new_calibration = copy.deepcopy(calibration)
        new_calibration.name = make_unique.by_number_at_end(
            new_calibration.name + " Copy", self.item_names
        )
        new_calibration.unique_id = model.Calibration.create_new_unique_id()
        return new_calibration

    def add(self, calibration):
        if any(c.unique_id == calibration.unique_id for c in self._calibrations):
            logger.warning(
                "Did not add calibration {} because it is already in the "
                "storage".format(calibration.name)
            )
            return
        self._calibrations.append(calibration)
        self._calibrations.sort(key=lambda c: c.name)

    def delete(self, calibration):
        self._calibrations.remove(calibration)
        self._delete_calibration_file(calibration)

    def _delete_calibration_file(self, calibration):
        try:
            os.remove(self._calibration_file_path(calibration))
        except FileNotFoundError:
            pass

    def rename(self, calibration, new_name):
        old_calibration_file_path = self._calibration_file_path(calibration)
        calibration.name = new_name
        new_calibration_file_path = self._calibration_file_path(calibration)
        try:
            os.rename(old_calibration_file_path, new_calibration_file_path)
        except FileNotFoundError:
            pass

    def get_first_or_none(self):
        if self._calibrations:
            return self._calibrations[0]
        else:
            return None

    def get_or_none(self, unique_id):
        try:
            return next(c for c in self._calibrations if c.unique_id == unique_id)
        except StopIteration:
            return None

    def _load_from_disk(self):
        try:
            # we sort because listdir sometimes returns files in weird order
            for file_name in sorted(os.listdir(self._calibration_folder)):
                if file_name.endswith(self._calibration_suffix):
                    self._load_calibration_from_file(file_name)
        except FileNotFoundError:
            pass
        self._load_recorded_calibrations()

    def _load_calibration_from_file(self, file_name):
        file_path = os.path.join(self._calibration_folder, file_name)
        calibration_tuple = self._load_data_from_file(file_path)
        if calibration_tuple:
            calibration = model.Calibration.from_tuple(calibration_tuple)
            if not self._from_same_recording(calibration):
                # the index range from another recording is useless and can lead
                # to confusion if it is rendered somewhere
                calibration.frame_index_range = [0, 0]
            self.add(calibration)

    def _load_recorded_calibrations(self):
        notifications = fm.load_pldata_file(self._rec_dir, "notify")
        for topic, data in zip(notifications.topics, notifications.data):
            if topic == "notify.calibration.calibration_data":
                try:
                    calib_result = model.CalibrationResult(
                        mapping_plugin_name=data["mapper_name"],
                        mapper_args=dict(data["mapper_args"]),
                    )
                except KeyError:
                    # notifications from old recordings will not have these fields!
                    continue
                mapping_method = "2d" if "2d" in data["calibration_method"] else "3d"
                # the unique id needs to be the same at every start or otherwise the
                # same calibrations would be added again and again. The timestamp is
                # the easiest datum that differs between calibrations but is the same
                # for every start
                unique_id = model.Calibration.create_unique_id_from_string(
                    str(data["timestamp"])
                )
                calibration = model.Calibration(
                    unique_id=unique_id,
                    name=make_unique.by_number_at_end(
                        "Recorded Calibration", self.item_names
                    ),
                    recording_uuid=self._recording_uuid,
                    mapping_method=mapping_method,
                    frame_index_range=self._get_recording_index_range(),
                    minimum_confidence=0.8,
                    is_offline_calibration=False,
                    result=calib_result,
                )
                self.add(calibration)

    def save_to_disk(self):
        os.makedirs(self._calibration_folder, exist_ok=True)
        calibrations_from_same_recording = (
            calib for calib in self._calibrations if self._from_same_recording(calib)
        )
        for calibration in calibrations_from_same_recording:
            self._save_data_to_file(
                self._calibration_file_path(calibration), calibration.as_tuple
            )

    def _from_same_recording(self, calibration):
        # There is a very similar, but public method in the CalibrationController.
        # This method only exists because its extremely inconvenient to access
        # controllers from storages and the logic is very simple.
        return calibration.recording_uuid == self._recording_uuid

    @property
    def items(self):
        return self._calibrations

    @property
    def item_names(self):
        return [calib.name for calib in self._calibrations]

    @property
    def _item_class(self):
        return model.Calibration

    @property
    def _calibration_folder(self):
        return os.path.join(self._rec_dir, "calibrations")

    def _calibration_file_name(self, calibration):
        file_name = "{}-{}.{}".format(
            calibration.name, calibration.unique_id, self._calibration_suffix
        )
        return self.get_valid_filename(file_name)

    def _calibration_file_path(self, calibration):
        return os.path.join(
            self._calibration_folder, self._calibration_file_name(calibration)
        )
