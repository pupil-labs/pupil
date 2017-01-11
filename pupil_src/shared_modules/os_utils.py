'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import platform, sys, os, time
from distutils.version import LooseVersion as VersionFormat
import subprocess as sp

import logging
logger = logging.getLogger(__name__)

os_name = platform.system()
if os_name == "Darwin":
    mac_version = VersionFormat(platform.mac_ver()[0])
    min_version = VersionFormat("10.11.0")

if os_name == "Darwin" and mac_version >= min_version:

    class Prevent_Idle_Sleep(object):

        def __init__(self):
            self.caffeine_process = None

        def __enter__(self):
            self.caffeine_process = sp.Popen(['caffeinate','-w', str(os.getpid())])
            logger.info('Disabled idle sleep.')

        def __exit__(self, type, value, traceback):
            if type is not None:
                pass # Exception occurred
            self.caffeine_process.terminate()
            self.caffeine_process = None
            logger.info('Re-enabled idle sleep.')
else:
    class Prevent_Idle_Sleep(object):

        def __init__(self):
            self.caffeine_process = None

        def __enter__(self):
            logger.info('Disabling idle sleep not supported on this OS version.')

        def __exit__(self, type, value, traceback):
            if type is not None:
                pass # Exception occurred
            pass


if __name__ == '__main__':
    with Prevent_Idle_Sleep():
        pass
