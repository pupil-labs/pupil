"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


def update_offline_calibrations_to_latest_version(rec_dir: str):
    from .calibration_storage_updater import CalibrationStorageUpdater

    CalibrationStorageUpdater.update_offline_calibrations_to_latest_version(rec_dir)
