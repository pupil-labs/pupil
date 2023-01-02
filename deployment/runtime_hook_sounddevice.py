"""Workaround to find sounddevice on Linux during runtime

See this issue for details:
https://github.com/spatialaudio/python-sounddevice/issues/130#issuecomment-1367883016
"""

import ctypes.util
import functools
import logging

logger = logging.getLogger(__name__)

logger.debug("Patching `ctypes.util.find_library` to find sounddevice...")
_find_library_original = ctypes.util.find_library


@functools.wraps(_find_library_original)
def _find_library_patched(name):
    if name == "portaudio":
        return "libportaudio.so.2"
    else:
        return _find_library_original(name)


ctypes.util.find_library = _find_library_patched

import sounddevice

logger.info("sounddevice import successful!")
logger.debug("Restoring original `ctypes.util.find_library`...")
ctypes.util.find_library = _find_library_original
del _find_library_patched
logger.debug("Original `ctypes.util.find_library` restored.")
