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

# TODO The square marker detection should return an object like this already. Also
# this object should offer a mean/centroid function to be used when drawing the
# marker toggle buttons
Square_Marker_Detection = collections.namedtuple(
    "Square_Marker_Detection", ["id", "id_confidence", "verts_px", "perimeter"]
)

from surface_tracker.surface_tracker_online import Surface_Tracker_Online
from surface_tracker.surface_tracker_offline import Surface_Tracker_Offline
