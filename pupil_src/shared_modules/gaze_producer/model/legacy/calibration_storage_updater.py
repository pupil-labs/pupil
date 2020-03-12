"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

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

        for calib_path in sorted(calib_dir.glob("*.plcal")):
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
                        cls.__update_and_save_calibration_v1_as_latest_verson(rec_dir, data)
                        continue  # Success
                except Exception as err:
                    logger.debug(str(err))

            # Failed to update
            logger.warning(f'Unable to update calibration "{calib_path.name}" from version {version} to the current version {Calibration.version}')

    @classmethod
    def __update_and_save_calibration_v1_as_latest_verson(cls, rec_dir, data):
        legacy_calibration = CalibrationV1.from_tuple(data)
        # TODO: Update and save only if an updated version doesn't already exist
        updated_calibration = legacy_calibration.updated()
        cls._save_calibration_to_file(rec_dir, updated_calibration)
