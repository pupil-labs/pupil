'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

'''
Video Capture provides the interface to get frames from diffferent backends.
Backends consist of a manager and at least one source class. The manager
is a Pupil plugin that provides an GUI that lists all available sources. The
source provides the stream of image frames.

These backends are available:
- UVC: Local USB sources
- NDSI: Remote Pupil Mobile sources
- Fake: Fallback, static grid image
- File: Loads video from file
'''

import os
import numpy as np
from glob import glob
from camera_models import load_intrinsics

import logging
logger = logging.getLogger(__name__)

from .base_backend import InitialisationError, StreamError, EndofVideoError
from .base_backend import Base_Source, Base_Manager
from .fake_backend import Fake_Source, Fake_Manager
from .file_backend import FileSeekError
from .file_backend import File_Source, File_Manager
from .uvc_backend import UVC_Source,  UVC_Manager

source_classes = [File_Source,  UVC_Source, Fake_Source]
manager_classes = [File_Manager, UVC_Manager, Fake_Manager]

try:
    from .ndsi_backend import NDSI_Source, NDSI_Manager
except ImportError:
    logger.info('Install pyndsi to use the Pupil Mobile backend')
else:
    source_classes.append(NDSI_Source)
    manager_classes.append(NDSI_Manager)

try:
    from .realsense_backend import Realsense_Source, Realsense_Manager
except ImportError:
    logger.info('Install pyrealsense to use the Intel RealSense backend')
else:
    source_classes.append(Realsense_Source)
    manager_classes.append(Realsense_Manager)


def init_playback_source(g_pool, source_path=None, *args, **kwargs):
    if source_path is None or os.path.splitext(source_path)[1] == '.fake':
        return Fake_Source(g_pool, source_path=source_path, *args, **kwargs)
    else:
        return File_Source(g_pool, source_path=source_path, *args, **kwargs)
