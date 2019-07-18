"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
from camera_models import Camera_Model
from video_capture.base_backend import Base_Source
import numpy as np


class Immutable_Capture:
    def __init__(self, capture: t.Type[Base_Source]):
        self.frame_size: t.Tuple[int, int] = (
            int(capture.frame_size[0]),
            int(capture.frame_size[1]),
        )
        self.intrinsics: t.Type[Camera_Model] = capture.intrinsics
        try:
            self.timestamps: np.ndarray = capture.timestamps
        except AttributeError:
            self.timestamps: np.ndarray = np.asarray([])
