"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
# isort: skip_file
# Import order matters for this module!

from gaze_producer.ui.select_and_refresh_menu import SelectAndRefreshMenu
from gaze_producer.ui.storage_edit_menu import StorageEditMenu

from gaze_producer.ui.calibration_menu import CalibrationMenu
from gaze_producer.ui.gaze_mapper_menu import GazeMapperMenu
from gaze_producer.ui.on_top_menu import OnTopMenu
from gaze_producer.ui.reference_location_menu import ReferenceLocationMenu

from gaze_producer.ui.reference_location_renderer import ReferenceLocationRenderer

from gaze_producer.ui.offline_calibration_timeline import OfflineCalibrationTimeline
from gaze_producer.ui.gaze_mapper_timeline import GazeMapperTimeline
from gaze_producer.ui.reference_location_timeline import ReferenceLocationTimeline
