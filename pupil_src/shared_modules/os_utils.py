'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import platform, sys, os, time, re
import subprocess as sp

import logging
logger = logging.getLogger(__name__)

os_name = platform.system()
if os_name == "Darwin":
    # version regex from - http://svn.python.org/projects/python/tags/r32rc1/Lib/distutils/version.py
    version_re = re.compile(r'^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$', re.VERBOSE)
    mac_version = platform.mac_ver()
    mac_version_str = mac_version[0]

    match = version_re.match(mac_version_str)
    if not match:
        logger.error("Invalid version string. Valid semantic versioning follows the MAJOR.MINOR.PATCH pattern with optional pre-release and pre-release build numbers.")

    try:
        (mac_major,mac_minor,mac_patch,mac_prerelease,mac_prerelease_num) = match.group(1,2,4,5,6) 
    except Exception as e:
        logger.error(e)


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
