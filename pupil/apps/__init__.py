"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
import sys

import pupil.utils.versions as version_utils


if getattr(sys, "frozen", False):
    version_file = os.path.join(sys._MEIPASS, "_version_string_")
else:
    version_file = None

app_version = version_utils.app_version(version_file)
