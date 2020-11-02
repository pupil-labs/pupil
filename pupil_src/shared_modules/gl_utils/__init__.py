"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .trackball import Trackball
from .utils import _Rectangle
from .utils import adjust_gl_view
from .utils import basic_gl_setup
from .utils import clear_gl_screen
from .utils import Coord_System
from .utils import cvmat_to_glmat
from .utils import get_content_scale
from .utils import get_framebuffer_scale
from .utils import get_monitor_workarea_rect
from .utils import get_window_frame_size_margins
from .utils import glClear
from .utils import glClearColor
from .utils import glFlush
from .utils import glViewport
from .utils import is_window_visible
from .utils import make_coord_system_norm_based
from .utils import make_coord_system_pixel_based
from .utils import window_coordinate_to_framebuffer_coordinate
from .window_position_manager import WindowPositionManager
