"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

try:
    from .methods import *
except ModuleNotFoundError:
    # when running from source compile cpp extension if necessary.
    from .build import build_cpp_extension
    build_cpp_extension()
    from .methods import *