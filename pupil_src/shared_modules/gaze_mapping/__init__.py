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
from .gazer_2d_v1x import Gazer2D_v1x

registered_gazer_classes = [Gazer2D_v1x]
registered_gazer_class_names = {
    cls.label: cls.__name__ for cls in registered_gazer_classes
}
