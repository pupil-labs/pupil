'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import platform, sys, os, time
os_name = platform.system()

# see https://github.com/google/python-subprocess32
if os.name == 'posix' and sys.version_info[0] < 3:
    try:
        import subprocess32 as sp
    except ImportError:
        import subprocess as sp
else:
    import subprocess as sp

import logging
logger = logging.getLogger(__name__)

caffeine_process = None

def disable_idle_sleep():
    if os_name == "Darwin":
        app_pid = os.getpid()
        # -w option stops `caffeinate` in case current app does not
        # terminate plugin correctly
        global caffeine_process
        caffeine_process = sp.Popen(['caffeinate', '-i', '-w', str(app_pid)])
        logger.info('Disabled idle sleep.')
    else:
        logger.info('Disabling idle sleep is not supported on this platform.')

def enable_idle_sleep():
    global caffeine_process
    if caffeine_process:
        caffeine_process.terminate()
        caffeine_process = None
        logger.info('Enabled idle sleep.')
