'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import platform, sys, os, time
import subprocess as sp

import logging
logger = logging.getLogger(__name__)

os_name = platform.system()
if os_name == "Darwin":
    mac_version = platform.mac_ver()
    mac_major,mac_minor,mac_patch = map(int,mac_version[0].split('.'))

if os_name == "Darwin" and mac_minor >=11:

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
