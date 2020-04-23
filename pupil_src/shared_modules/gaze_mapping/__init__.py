"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from . import matching
from . import gazer_base
from .gazer_base import CalibrationError
from .gazer_2d_v1x import Gazer2D_v1x
from .gazer_2d_v2 import Gazer2D
from .gazer_3d_v1x import Gazer3D_v1x
from .gazer_hmd_v1x import GazerHMD3D_v1x


def registered_gazer_classes() -> list:
    return gazer_base.GazerBase.registered_gazer_classes()


def registered_gazer_labels_by_class_names() -> dict:
    return {cls.__name__: cls.label for cls in registered_gazer_classes()}


def registered_gazer_classes_by_class_name() -> dict:
    return {cls.__name__: cls for cls in registered_gazer_classes()}


default_gazer_class = Gazer3D_v1x
assert default_gazer_class in registered_gazer_classes()
