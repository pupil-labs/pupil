'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

#logging
import logging
logger = logging.getLogger(__name__)


# import detector classes from sibling files
from screen_marker_calibration import Screen_Marker_Calibration
from manual_marker_calibration import Manual_Marker_Calibration
from natural_features_calibration import Natural_Features_Calibration
from camera_intrinsics_estimation import Camera_Intrinsics_Estimation
from accuracy_test import Accuracy_Test
from gaze_mappers import Dummy_Gaze_Mapper, Simple_Gaze_Mapper, Volumetric_Gaze_Mapper,Bilateral_Gaze_Mapper

calibration_plugins =  [Screen_Marker_Calibration,
                        Manual_Marker_Calibration,
                        Natural_Features_Calibration,
                        Camera_Intrinsics_Estimation,
                        Accuracy_Test ]

gaze_mapping_plugins = [Dummy_Gaze_Mapper,
                        Simple_Gaze_Mapper,
                        Volumetric_Gaze_Mapper,
                        Bilateral_Gaze_Mapper]
