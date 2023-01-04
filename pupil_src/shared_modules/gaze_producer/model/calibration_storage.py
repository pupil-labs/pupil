"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import copy
import logging
import os
import pathlib
from collections import namedtuple

import file_methods as fm
import make_unique
from gaze_mapping import (
    default_gazer_class,
    gazer_classes_by_class_name,
    registered_gazer_classes,
    user_selectable_gazer_classes,
    user_selectable_gazer_classes_posthoc,
)
from gaze_mapping.notifications import (
    CalibrationResultNotification,
    CalibrationSetupNotification,
)
from gaze_producer import model
from observable import Observable
from storage import Storage

logger = logging.getLogger(__name__)


class CalibrationStorage(Storage, Observable):
    _calibration_suffix = "plcal"

    def __init__(self, rec_dir, get_recording_index_range, recording_uuid):
        self._rec_dir = rec_dir
        self._get_recording_index_range = get_recording_index_range
        self._recording_uuid = str(recording_uuid)
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
            gazer_class_name=default_gazer_class.__name__,
            frame_index_range=self._get_recording_index_range(),
            minimum_confidence=0.8,
            is_offline_calibration=True,
            status="Not calculated yet",
        )

    def __create_prerecorded_calibration(
        self, number: int, result_notification: CalibrationResultNotification
    ):
        timestamp = result_notification.timestamp

        # the unique id needs to be the same at every start or otherwise the
        # same calibrations would be added again and again. The timestamp is
        # the easiest datum that differs between calibrations but is the same
        # for every start
        unique_id = model.Calibration.create_unique_id_from_string(str(timestamp))

        name = "Recorded Calibration"
        if number > 1:
            name += f" {number}"

        return model.Calibration(
            unique_id=unique_id,
            name=name,
            recording_uuid=self._recording_uuid,
            gazer_class_name=result_notification.gazer_class_name,
            frame_index_range=self._get_recording_index_range(),
            minimum_confidence=0.8,
            is_offline_calibration=False,
            status="Not calculated yet",
            calib_params=result_notification.params,
        )

    def duplicate_calibration(self, calibration):
        new_calibration = copy.deepcopy(calibration)
        new_calibration.name = make_unique.by_number_at_end(
            new_calibration.name + " Copy", self.item_names
        )
        new_calibration.unique_id = model.Calibration.create_new_unique_id()
        return new_calibration

    def add(self, calibration, overwrite=False):
        for calib in self._calibrations.copy():
            if calib.unique_id != calibration.unique_id:
                continue
            if overwrite:
                self._calibrations.remove(calib)
                break
            logger.warning(
                f"Did not add calibration {calibration.name} ({calibration.unique_id})"
                " because it is already in the storage. Currently in storage:\n"
                + "\n".join(f"- {c.name} ({c.unique_id})" for c in self._calibrations)
            )
            return
        is_calib_editable = (
            self._from_same_recording(calibration)
            and calibration.is_offline_calibration
        )
        if is_calib_editable:
            available_gazer_classes = user_selectable_gazer_classes_posthoc()
        else:
            available_gazer_classes = registered_gazer_classes()
        gazer_class_names = gazer_classes_by_class_name(available_gazer_classes).keys()
        if calibration.gazer_class_name not in gazer_class_names:
            logger.warning(
                f"Did not add calibration {calibration.name} ({calibration.unique_id}) because gaze mapping method ({calibration.gazer_class_name}) is not available."
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
            folder_path = pathlib.Path(self._calibration_folder)
            # pattern `[!.]`: exclude OS-specific meta-data files
            pattern = f"[!.]*{self._calibration_suffix}"
            for file_name in sorted(folder_path.glob(pattern)):
                self._load_calibration_from_file(file_name)
        except FileNotFoundError:
            pass
        self._load_recorded_calibrations()

    def _load_calibration_from_file(self, file_name):
        file_path = os.path.join(self._calibration_folder, file_name)
        calibration_dict = self._load_data_from_file(file_path)
        if not calibration_dict:
            return
        try:
            calibration = model.Calibration.from_dict(calibration_dict)
        except ValueError as err:
            logger.debug(str(err))
            return
        if not self._from_same_recording(calibration):
            # the index range from another recording is useless and can lead
            # to confusion if it is rendered somewhere
            calibration.frame_index_range = [0, 0]
        self.add(calibration)

    def _load_recorded_calibrations(self):
        notifications = fm.load_pldata_file(self._rec_dir, "notify")
        counter = 1
        for topic, data in zip(notifications.topics, notifications.data):
            if topic.startswith("notify."):
                # Remove "notify." prefix
                data = data._deep_copy_dict()
                data["subject"] = data["topic"][len("notify.") :]
                del data["topic"]
            else:
                continue
            if (
                CalibrationResultNotification.calibration_format_version()
                != model.Calibration.version
            ):
                logger.debug(
                    f"Must update CalibrationResultNotification to match Calibration version"
                )
                continue
            try:
                note = CalibrationResultNotification.from_dict(data)
            except ValueError as err:
                logger.debug(str(err))
                continue
            calibration = self.__create_prerecorded_calibration(
                number=counter, result_notification=note
            )
            self.add(calibration, overwrite=True)
            counter += 1

    def save_to_disk(self):
        os.makedirs(self._calibration_folder, exist_ok=True)
        calibrations_from_same_recording = (
            calib for calib in self._calibrations if self._from_same_recording(calib)
        )
        for calibration in calibrations_from_same_recording:
            self._save_calibration_to_file(self._rec_dir, calibration)

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
        return type(self).__calibration_model_class()

    @property
    def _calibration_folder(self):
        # TODO: Backwards compatibility; remove in favor of class method
        return str(self._calibration_directory_from_recording(self._rec_dir))

    def _calibration_file_name(self, calibration):
        # TODO: Backwards compatibility; remove in favor of class method
        return type(self).__calibration_file_name(calibration)

    def _calibration_file_path(self, calibration):
        # TODO: Backwards compatibility; remove in favor of class method
        return str(
            self.__calibration_file_path_in_recording(self._rec_dir, calibration)
        )

    ### Private

    @staticmethod
    def __calibration_model_class():
        return model.Calibration

    @staticmethod
    def _calibration_directory_from_recording(rec_dir) -> pathlib.Path:
        return pathlib.Path(rec_dir).joinpath("calibrations")

    @classmethod
    def __calibration_file_name(cls, calibration) -> str:
        file_name = (
            f"{calibration.name}-{calibration.unique_id}.{cls._calibration_suffix}"
        )
        return cls.get_valid_filename(file_name)

    @classmethod
    def __calibration_file_path_in_recording(cls, rec_dir, calibration) -> pathlib.Path:
        file_name = cls.__calibration_file_name(calibration)
        calib_dir = cls._calibration_directory_from_recording(rec_dir)
        return calib_dir.joinpath(file_name)

    @classmethod
    def _save_calibration_to_file(
        cls, rec_dir, calibration: model.Calibration, overwrite_if_exists=True
    ):
        # Sanity check
        assert cls.__calibration_model_class() == calibration.__class__

        # Get all properties needed to save calibration
        version = calibration.__class__.version
        data = calibration.as_dict
        path = str(cls.__calibration_file_path_in_recording(rec_dir, calibration))

        # Save a dictionary representation of the calibration
        dict_representation = {"version": version, "data": data}
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.isfile(path):
            if not overwrite_if_exists:
                logger.debug(
                    f'Calibration file already exists, won\'t overwrite. "{path}"'
                )
                return
            else:
                logger.debug(
                    f'Calibration file already exists, overwriting new version. "{path}"'
                )

        fm.save_object(dict_representation, path)
