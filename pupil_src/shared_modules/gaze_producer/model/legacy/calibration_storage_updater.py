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
import typing as T

import file_methods as fm
from gaze_producer.model.calibration import Calibration
from gaze_producer.model.calibration_storage import CalibrationStorage
from pupil_recording import PupilRecording

from .calibration_v1 import CalibrationV1

logger = logging.getLogger(__name__)


class CalibrationStorageUpdater(CalibrationStorage):
    @classmethod
    def update_offline_calibrations_to_latest_version(cls, rec_dir):
        calib_dir = cls._calibration_directory_from_recording(rec_dir)

        if not calib_dir.exists():
            return

        if not calib_dir.is_dir():
            return  # TODO: Raise exception - "calibrations" must be a directory

        for calib_path in sorted(calib_dir.glob("[!.]*.plcal")):
            calib_dict = fm.load_object(calib_path)
            version = calib_dict.get("version", None)
            data = calib_dict.get("data", None)

            if version == Calibration.version:
                # Calibration already at current version
                continue  # No-op

            elif version > Calibration.version:
                # Calibration at newer version than current model
                continue  # No-op

            elif data is not None:
                try:
                    if version == CalibrationV1.version == 1:
                        cls.__update_and_save_calibration_v1_as_latest_version(
                            rec_dir, data
                        )
                        continue  # Success
                except Exception as err:
                    logger.warning(str(err))

            # Failed to update
            logger.warning(
                f'Unable to update calibration "{calib_path.name}" from version {version} to the current version {Calibration.version}'
            )

    @classmethod
    def __update_and_save_calibration_v1_as_latest_version(cls, rec_dir, data):
        legacy_calibration = CalibrationV1.from_tuple(data)

        recording_uuid = str(PupilRecording(rec_dir).meta_info.recording_uuid)
        is_imported = legacy_calibration.recording_uuid != recording_uuid
        if is_imported:
            raise ValueError(
                "Updating imported (read-only) calibrations is not supported. "
                f"{legacy_calibration.name}"
            )

        updated_calibration = legacy_calibration.updated()
        cls._save_calibration_to_file(
            rec_dir, updated_calibration, overwrite_if_exists=False
        )
