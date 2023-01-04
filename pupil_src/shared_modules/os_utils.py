"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os
import platform
import subprocess as sp
import sys
import time
import traceback

from version_utils import parse_version

logger = logging.getLogger(__name__)

os_name = platform.system()
if os_name == "Darwin":
    mac_version = parse_version(platform.mac_ver()[0])
    min_version = parse_version("10.11.0")

if os_name == "Darwin" and mac_version >= min_version:

    class Prevent_Idle_Sleep:
        def __init__(self):
            self.caffeine_process = None

        def __enter__(self):
            self.caffeine_process = sp.Popen(["caffeinate", "-w", str(os.getpid())])
            logger.debug("Disabled idle sleep.")

        def __exit__(self, etype, value, tb):
            if etype is not None:
                logger.debug("".join(traceback.format_exception(etype, value, tb)))
            self.caffeine_process.terminate()
            self.caffeine_process = None
            # NOTE: Suppress KeyboardInterrupt
            return etype is KeyboardInterrupt

else:

    class Prevent_Idle_Sleep:
        def __init__(self):
            self.caffeine_process = None

        def __enter__(self):
            logger.debug("Disabling idle sleep not supported on this OS version.")

        def __exit__(self, etype, value, tb):
            if etype is not None:
                logger.debug("".join(traceback.format_exception(etype, value, tb)))
            # NOTE: Suppress KeyboardInterrupt
            return etype is KeyboardInterrupt


def patch_pyre_zhelper_cdll():
    """Fixes https://github.com/pupil-labs/pupil/issues/1919

    When running the v2.0 bundle on macOS 10.14, `ctypes.CDLL("libSystem.dylib")` fails
    to load which is required by pyre.zhelper. `libSystem.dylib` is not part of the
    bundle on purpose, as `ctypes.CDLL` is usually able to fallback to the system-
    provided library at `/usr/lib/libSystem.dylib`.

    The fallback mechanism is provided by the $DYLD_FALLBACK_LIBRARY_PATH variable.
    PyInstaller makes use of it to inject the bundle path to prioritize bundled
    libraries over system libraries:
    https://github.com/pyinstaller/pyinstaller/blob/v3.6/PyInstaller/loader/pyiboot01_bootstrap.py#L203

    Upon inspection, `/usr/lib` is correcly present in $DYLD_FALLBACK_LIBRARY_PATH when
    running the macOS bundle on macOS 10.14. This indicates an underlying deeper issue.

    Curiously, `ctypes.util.find_library("libSystem.dylib")` works as expected and
    returns `/usr/lib/libSystem.dylib` when running the above setup. This patch makes
    use of that fact by implementing an additional fallback that attempts to load the
    initially missing library based on its absolute path.

    As `pyre.zhelper` is the only place calling `ctypes.CDLL("libSystem.dylib")`, we
    will only patch this specific CDLL instance.

    This patch will only be applied if the problematic setup is detected.
    """
    running_from_bundle = getattr(sys, "frozen", False)
    if os_name != "Darwin" or not running_from_bundle:
        return

    import ctypes.util

    import pyre.zhelper

    class AbsolutePathFallbackCDLL(ctypes.CDLL):
        def __init__(self, name, *args, **kwargs):
            try:
                super().__init__(name, *args, **kwargs)
            except Exception:
                abs_library_path = ctypes.util.find_library(name)
                if abs_library_path is not None:
                    super().__init__(abs_library_path, *args, **kwargs)
                else:
                    raise

    pyre.zhelper.CDLL = AbsolutePathFallbackCDLL


if __name__ == "__main__":
    with Prevent_Idle_Sleep():
        pass
