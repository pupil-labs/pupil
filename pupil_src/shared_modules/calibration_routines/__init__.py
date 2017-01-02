'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


# import detector classes from sibling files
from screen_marker_calibration import Screen_Marker_Calibration
from manual_marker_calibration import Manual_Marker_Calibration
from natural_features_calibration import Natural_Features_Calibration
from camera_intrinsics_estimation import Camera_Intrinsics_Estimation
from adjust_calibration import Adjust_Calibration
from accuracy_test import Accuracy_Test
from hmd_calibration import HMD_Calibration
from gaze_mappers import Dummy_Gaze_Mapper, Monocular_Gaze_Mapper, Binocular_Gaze_Mapper,Vector_Gaze_Mapper,Binocular_Vector_Gaze_Mapper,Dual_Monocular_Gaze_Mapper

calibration_plugins =  [Screen_Marker_Calibration,
                        Manual_Marker_Calibration,
                        Natural_Features_Calibration,
                        Camera_Intrinsics_Estimation,
                        Accuracy_Test,
                        Adjust_Calibration,
                        HMD_Calibration ]

gaze_mapping_plugins = [Dummy_Gaze_Mapper,
                        Monocular_Gaze_Mapper,
                        Vector_Gaze_Mapper,
                        Binocular_Gaze_Mapper,
                        Binocular_Vector_Gaze_Mapper,
                        Dual_Monocular_Gaze_Mapper]
