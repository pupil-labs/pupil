"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
from enum import Enum

Marker = collections.namedtuple("Marker", ["id", "id_confidence", "verts", "perimeter"])


class Heatmap_Mode(Enum):
    WITHIN_SURFACE = "Gaze within each surface"
    ACROSS_SURFACES = "Gaze across different surfaces"


from surface_tracker_future.surface_tracker_online import Surface_Tracker_Online_Future
from surface_tracker_future.surface_tracker_offline import (
    Surface_Tracker_Offline_Future
)
