"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from typing import Dict, List, Type

from . import gazer_base, matching
from .gazer_2d import Gazer2D
from .gazer_3d import Gazer3D, GazerHMD3D, NeonGazer3D, PosthocGazerHMD3D
from .gazer_base import CalibrationError


def registered_gazer_classes() -> List[Type[gazer_base.GazerBase]]:
    return gazer_base.GazerBase.registered_gazer_classes()


def user_selectable_gazer_classes() -> List[Type[gazer_base.GazerBase]]:
    gazers = registered_gazer_classes()
    gazers = filter(lambda g: g not in (GazerHMD3D, PosthocGazerHMD3D), gazers)
    return list(gazers)


def user_selectable_gazer_classes_posthoc() -> list:
    gazers = registered_gazer_classes()
    gazers = filter(lambda g: g not in (GazerHMD3D,), gazers)
    return list(gazers)


def gazer_labels_by_class_names(
    gazers: List[Type[gazer_base.GazerBase]],
) -> Dict[str, str]:
    return {cls.__name__: cls.label for cls in gazers}


def gazer_classes_by_class_name(
    gazers: List[Type[gazer_base.GazerBase]],
) -> Dict[str, Type[gazer_base.GazerBase]]:
    return {cls.__name__: cls for cls in gazers}


default_gazer_class = Gazer3D
assert default_gazer_class in registered_gazer_classes()
