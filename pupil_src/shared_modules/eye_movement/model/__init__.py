"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .color import (
    Color,
    Color_RGB,
    Color_RGBA,
    Color_Palette,
    Base_Color_Palette,
    Defo_Color_Palette,
)
from .immutable_capture import Immutable_Capture
from .segment import (
    Segment_Class,
    Segment_Base_Type,
    Classified_Segment,
    Classified_Segment_Factory,
)
from .storage import Classified_Segment_Storage
from .time_range import Time_Range
