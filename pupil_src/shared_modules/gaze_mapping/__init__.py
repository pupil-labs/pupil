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
from .gazer_2d import Gazer2D
from .gazer_3d import Gazer3D, GazerHMD3D


def registered_gazer_classes() -> list:
    return gazer_base.GazerBase.registered_gazer_classes()


def registered_gazer_labels_by_class_names() -> dict:
    return {cls.__name__: cls.label for cls in registered_gazer_classes()}


def registered_gazer_classes_by_class_name() -> dict:
    return {cls.__name__: cls for cls in registered_gazer_classes()}


default_gazer_class = Gazer3D
assert default_gazer_class in registered_gazer_classes()
