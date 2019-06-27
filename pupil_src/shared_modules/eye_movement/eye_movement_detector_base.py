"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from plugin import Analysis_Plugin_Base


class Eye_Movement_Detector_Base(Analysis_Plugin_Base):
    icon_chr = chr(0xEC03)
    icon_font = "pupil_icons"
