"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2021 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import ctypes.util
import functools


print("Attempting to import OpenGL using patched `ctypes.util.find_library`...")
_find_library_original = ctypes.util.find_library


@functools.wraps(_find_library_original)
def _find_library_patched(name):
    if name == "OpenGL":
        return "/System/Library/Frameworks/OpenGL.framework/OpenGL"
    else:
        return _find_library_original(name)


ctypes.util.find_library = _find_library_patched

import OpenGL.GL

print("OpenGL import successful!")
print("Restoring original `ctypes.util.find_library`...")
ctypes.util.find_library = _find_library_original
del _find_library_patched
print("Original `ctypes.util.find_library` restored.")
