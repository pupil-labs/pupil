"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .menu_content import Menu_Content
from .navigation_buttons import Prev_Segment_Button, Next_Segment_Button
from .segment_overlay import (
    color_from_segment,
    Segment_Overlay_Renderer,
    Segment_Overlay_Image_Renderer,
    Segment_Overlay_GL_Context_Renderer,
)
