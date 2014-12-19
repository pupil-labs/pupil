'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

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

detector_by_index =  (   Screen_Marker_Calibration,
                        Manual_Marker_Calibration,
                        Natural_Features_Calibration,
                        Camera_Intrinsics_Estimation,
                        Accuracy_Test )

name_by_index = [p.__name__ for p in detector_by_index]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
detector_by_name = dict(zip(name_by_index,detector_by_index))