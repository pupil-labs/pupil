'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

# when running from source compile cpp extension if necessary.
import sys
if not getattr(sys, 'frozen', False):
    from .build import build_cpp_extension
    build_cpp_extension()

from .detector_2d import Detector_2D
from .detector_3d import Detector_3D


#explicit import here for pyinstaller because it will not search .pyx source files.
from .visualizer_3d import Eye_Visualizer
