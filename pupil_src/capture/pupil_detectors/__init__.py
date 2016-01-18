'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from canny_detector import Canny_Detector
from detector_2d import Detector_2D
from detector_3d import Detector_3D

import sys
if not getattr(sys, 'frozen', False):
    from build import build_cpp_extension
    build_cpp_extension()

#explict import here for pyinstaller because it will not search .pyx source files.
import visualizer_3d

