"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os

broken_data = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/broken/eye0.mp4"
)
multiple_data = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/multiple/eye0.mp4"
)
single_data = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/single/eye0.mp4"
)
